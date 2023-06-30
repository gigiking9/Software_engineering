from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class PositionEmbedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x):  # 上一层一般就是embedding层，batch_size,seq_len,model_dim
        if self.size is None or self.mode == 'sum':
            self.size = int(x.shape[-1])  # d_model的长度，例如512
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]  #

        # 生成位置编码矩阵
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)  # (1, size/2)

        # 生成位置序列
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # 生成0~seq_len-1的序列
        position_i = K.expand_dims(position_i, 2)  # (batch_size, seq_len, 1)
        position_ij = K.dot(position_i, position_j)  # (batch_size, seq_len, size/2)

        # 使用正弦和余弦函数对位置编码进行交叉拼接
        position_ij_2i = K.sin(position_ij)[..., tf.newaxis]  # (batch_size, seq_len, size/2, 1)
        position_ij_2i_1 = K.cos(position_ij)[..., tf.newaxis]  # (batch_size, seq_len, size/2, 1)
        position_ij = K.concatenate([position_ij_2i, position_ij_2i_1], axis=-1)  # (batch_size, seq_len, size/2, 2)
        position_ij = K.reshape(position_ij, (batch_size, seq_len, self.size))  # (batch_size, seq_len, size)

        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


'''
query = tf.random.truncated_normal([100, 50, 150])
w = Position_Embedding(150,'concat')(query)
print(w.shape)
'''