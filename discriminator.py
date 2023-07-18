import tensorflow as tf

from resnet import ResBlock


class BuildDiscriminator(tf.keras.Model):
    def __init__(self, layer_dim, seq_len):
        super(BuildDiscriminator, self).__init__()
        dim = layer_dim
        self.dim = layer_dim
        self.seq_len = seq_len

        self.block = tf.keras.Sequential([
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        ])
        self.conv1d = tf.keras.layers.Conv1D(dim, 16, 1, padding='valid')
        self.linear = tf.keras.layers.Dense(seq_len * dim, activation='linear')

    def call(self, input, **kwargs):
        output = tf.transpose(input, [0, 2, 1])
        output = self.conv1d(output)
        output = self.block(output)
        output = tf.reshape(output, (-1, 64, 4))
        output = self.linear(output)
        return output
