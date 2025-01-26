'''
This script is used to convert tensorflow model to onnx model
Author: Radmir Kadyrov
'''

import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.models.load_model('weights/model.h5')

input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='digit')]

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
onnx.save(onnx_model, 'weights/model.onnx')

