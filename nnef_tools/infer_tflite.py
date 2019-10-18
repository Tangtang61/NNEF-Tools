#!/usr/bin/env python

# Copyright (c) 2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function, absolute_import

import sys

# Python2: Ensure that load from current directory is enabled, but load from the directory of the script is disabled
if len(sys.path) == 0:
    sys.path.append('')
if sys.path[0] != '':
    sys.path[0] = ''

import argparse
import os
import tempfile
import shutil

import tensorflow as tf
import numpy as np
import nnef

from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.conversion.tensorflow import nnef_to_tf
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
from nnef_tools.io.tensorflow import tflite_io
from nnef_tools.optimization.tensorflow.tf_data_format_optimizer import Optimizer as TFDataFormatOptimizer
from nnef_tools.optimization.data_format_optimizer import IOTransform


def get_args():
    parser = argparse.ArgumentParser(description="NNEF inference tool with Tensorflow Lite backend",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="""Tips:
- If you refer to a Python package or module that is not in the current directory,
please add its location to PYTHONPATH.
- Quote parameters if they contain spaces or special characters.
""")

    parser.add_argument("network",
                        help="Path of NNEF file, directory or archive")

    parser.add_argument('--input',
                        required=False,
                        nargs='+',
                        help="Path of input tensors.\n"
                             "By default they are read from the standard input through a pipe or redirect.")

    parser.add_argument('--output',
                        required=False,
                        help="The path of the output directory.\n"
                             "By default the standard output is used, but only if the command is piped or redirected.")

    args = parser.parse_args()

    has_weights = not (os.path.isfile(args.network) and not args.network.endswith('.tgz'))
    if not has_weights:
        raise utils.NNEFToolsException("Error: Seems like you have specified an NNEF file without weights. "
                                       "Please use generate_weights.py")

    return args


def run_tflite_model(model_path, inputs_np):
    if not isinstance(inputs_np, (list, tuple)):
        inputs_np = (inputs_np,)
    interpreter = (tf.lite.Interpreter if hasattr(tf, "lite") else tf.contrib.lite.Interpreter)(model_path=model_path)
    interpreter.allocate_tensors()
    for input_detail, input_np in zip(interpreter.get_input_details(), inputs_np):
        interpreter.set_tensor(input_detail['index'], input_np)
    interpreter.invoke()
    return tuple(interpreter.get_tensor(output_detail['index']) for output_detail in interpreter.get_output_details())


def transform_input(input_np):
    input_np = input_np[:1]  # HACK

    rank = len(input_np.shape)
    if rank >= 3:
        return np.transpose(input_np, (0,) + tuple(range(2, rank)) + (1,))
    return input_np


def transform_output(output_np):
    rank = len(output_np.shape)
    if rank >= 3:
        return np.transpose(output_np, (0, rank - 1) + tuple(range(1, rank - 1)))
    return output_np


def main():
    args = get_args()

    if args.input is None:
        if sys.stdin.isatty():
            raise utils.NNEFToolsException("No input provided!")
        utils.set_stdin_to_binary()

    if args.output is None:
        if sys.stdout.isatty():
            raise utils.NNEFToolsException("No output provided!")
        utils.set_stdout_to_binary()

    nnef_graph = nnef_io.Reader([NNEFParserConfig.STANDARD_CONFIG, nnef_to_tf.ParserConfig])(args.network)
    input_names = [t.name for t in nnef_graph.inputs]
    output_names = [t.name for t in nnef_graph.outputs]

    tf_graph, _ = nnef_to_tf.Converter()(nnef_graph)
    del nnef_graph

    TFDataFormatOptimizer(io_transform=IOTransform.SMART_NCHW_TO_NHWC, merge_transforms_into_variables=True)(tf_graph)

    dir = tempfile.mkdtemp(prefix="nnef_")

    try:
        tflite_path = os.path.join(dir, "network.tflite")
        tflite_io.Writer(convert_from_tf_py=True)(tf_graph, tflite_path)
        del tf_graph

        if args.input is None:
            inputs = tuple(nnef.read_tensor(sys.stdin)[0] for _ in range(len(input_names)))
        elif len(args.input) == 1 and os.path.isdir(args.input[0]):
            inputs = tuple(nnef_io.read_nnef_tensor(os.path.join(args.input[0], input_name + '.dat'))
                           for input_name in input_names)
        else:
            inputs = tuple(nnef_io.read_nnef_tensor(path) for path in args.input)

        outputs = run_tflite_model(tflite_path, tuple(transform_input(input) for input in inputs))
        del inputs

        if args.output is None:
            for output in outputs:
                nnef.write_tensor(sys.stdout, output)
        else:
            for output_name, output in zip(output_names, outputs):
                output_path = os.path.join(args.output, output_name + '.dat')
                nnef_io.write_nnef_tensor(output_path, transform_output(output))
    finally:
        shutil.rmtree(dir)


if __name__ == '__main__':
    main()

# Tested on networks/inception_v2.nnef
