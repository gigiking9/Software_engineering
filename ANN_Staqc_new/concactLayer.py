import os
import numpy as np
import random
import tensorflow as tf
from keras.layers import Layer
from keras import backend as K

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

class ConcatLayer(Layer):
    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConcatLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # 沿着第二个轴拆分输入张量
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        # 沿着第三个轴连接拆分后的张量
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        # 去除尺寸为1的维度
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        print(block_level_code_output)
        return block_level_code_output

    def compute_output_shape(self, input_shape):
        print("===========================", input_shape)
        return (input_shape[0], input_shape[1] * input_shape[2])
