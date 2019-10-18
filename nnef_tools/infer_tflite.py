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

    #     parser.add_argument("--output-names",
    #                         nargs='*',
    #                         help="""List tensor names to ensure that those tensors are exported.
    # If this option is not specified the graph's output tensors are exported.
    # --output-names: Export nothing
    # --output-names a b c: Export the tensors a, b and c
    # --output-names '*': Export all activation tensors
    #  """)

    #     parser.add_argument("--device",
    #                         required=False,
    #                         help="""Set device: cpu, cuda, cuda:0, cuda:1, etc.
    # Default: cuda if available, cpu otherwise.""")

    args = parser.parse_args()

    has_weights = not (os.path.isfile(args.network) and not args.network.endswith('.tgz'))
    if not has_weights:
        raise utils.NNEFToolsException("Error: Seems like you have specified an NNEF file without weights. "
                                       "Please use generate_weights.py")

    return args


def run_tflite_model(model_path, input_np):
    interpreter = (tf.lite.Interpreter if hasattr(tf, "lite") else tf.contrib.lite.Interpreter)(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_np)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])


def main():
    args = get_args()
    nnef_graph = nnef_io.Reader([NNEFParserConfig.STANDARD_CONFIG, nnef_to_tf.ParserConfig])(args.network)
    tf_graph, _ = nnef_to_tf.Converter()(nnef_graph)
    TFDataFormatOptimizer(io_transform=IOTransform.SMART_NCHW_TO_NHWC, merge_transforms_into_variables=True)(tf_graph)
    dir = tempfile.mkdtemp(prefix="nnef_")
    try:
        tflite_path = os.path.join(dir, "network.tflite")
        tflite_io.Writer(convert_from_tf_py=True)(tf_graph, tflite_path)
        assert len(args.input) == 1
        input_np = np.transpose(nnef_io.read_nnef_tensor(args.input[0]), (0, 2, 3, 1))[2:3]
        output = run_tflite_model(tflite_path, input_np)
        print(np.argmax(output, axis=1))
    finally:
        shutil.rmtree(dir)


if __name__ == '__main__':
    main()
