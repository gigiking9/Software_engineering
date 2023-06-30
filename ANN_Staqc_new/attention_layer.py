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

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('AttentionLayer 需要接收两个输入的列表形式.')

        if not input_shape[0][2] == input_shape[1][2]:
            raise ValueError('嵌入大小应该相同.')

        self.kernel = self.add_weight(
            shape=(input_shape[0][2], input_shape[0][2]),
            initializer='glorot_uniform',
            name='kernel',
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 执行注意力机制计算
        a = K.dot(inputs[0],
                  self.kernel)  # (batch_size, seq_len, embedding_dim) * (embedding_dim, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        y_trans = K.permute_dimensions(inputs[1],
                                       (0, 2, 1))  # 将输入 inputs[1] 的维度进行转置以适应 batch_dot 的计算
        b = K.batch_dot(a, y_trans, axes=[2,
                                          1])  # (batch_size, seq_len, embedding_dim) * (batch_size, embedding_dim, seq_len) -> (batch_size, seq_len, seq_len)
        return K.tanh(b)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], input_shape[1][1])
