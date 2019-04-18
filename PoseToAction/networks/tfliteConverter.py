# -*- coding: utf-8 -*-
# @Time    : 18-7-16 上午10:26
# @Author  : edvard_hua@live.com
# @FileName: gen_tflite_coreml.py
# @Software: PyCharm

import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf


input_node_name = "input_layer"
output_node_name = "output_layer"
output_filename = "model.tflite"
path = "/home/yash/Desktop/"
frozen_pb = path + "model.pb"
output_path = path
converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
    frozen_pb,
    [input_node_name],
    [output_node_name]
)
tflite_model = converter.convert()
open(os.path.join(output_path, output_filename), "wb").write(tflite_model)
print("Generate tflite success.")

