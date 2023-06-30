from __future__ import print_function
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        self.epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化可学习的权重：beta 和 gamma
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zero',
                                    name='beta')
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='one',
                                     name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        # 计算最后一个轴上的均值和方差
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        # 对输入进行归一化
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        # 应用 gamma 和 beta 到归一化后的输入上
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
