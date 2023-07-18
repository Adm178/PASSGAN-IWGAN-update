import tensorflow as tf
from resnet import ResBlock


class BuildGenerator(tf.keras.Model):
    def __init__(self, layer_dim, seq_len):
        super(BuildGenerator, self).__init__()
        dim = layer_dim
        self.dim = layer_dim
        self.seq_len = seq_len

        self.fc1 = tf.keras.layers.Dense(128, activation='linear', input_shape=(dim * seq_len,))
        self.block = tf.keras.Sequential([
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        ])
        self.conv1 = tf.keras.layers.Conv1D(8, 1, 1, padding='valid')
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, noise, **kwargs):
        output = self.fc1(noise)
        output = tf.reshape(output, (-1, 2, 128))
        output = self.block(output)
        output = tf.reshape(output, [1, 32, 64])
        output = self.conv1(output)
        output = tf.transpose(output, [0, 2, 1])
        output = self.softmax(output)
        return tf.reshape(output, [16, 1, 16])
