import tensorflow as tf


class ResBlock(tf.keras.Model):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = tf.keras.Sequential([
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 8, padding='same'),
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 8, padding='same'),
        ])

    def call(self, input, **kwargs):
        output = self.res_block(input)
        return input + (0.3 * output)
