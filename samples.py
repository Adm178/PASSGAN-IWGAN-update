from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.summary import summary
from tensorflow.python.tools import saved_model_utils

keras.backend.clear_session()
output = []
offset = 95
model_dir = 'models/generator/rock_you20200523-130754'
log_dir = 'models/generator/'
tag_set = 'serve'

available_model = tf.saved_model.contains_saved_model(model_dir)
print(available_model)

model = tf.saved_model.load(model_dir)
for i in range(200):
    z = tf.constant(tf.random.normal([2, 1, 32], dtype=tf.dtypes.float32))
    samples = model(z, training=False)
    samples = np.argmax(samples, axis=2)
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append([samples[i][j]])
        decoded = list(np.asarray(decoded) + offset)
        output.append(tuple(decoded))

y = [i[0] for i in output]
charList = [chr(y[i]) for i in range(0, len(output))]
with open('data/samples.txt', "w") as f:
    writer = csv.writer(f, delimiter="'", lineterminator="\r\n")
    writer.writerows(charList)

print('DISCLAIMER: The following generated characters are for research purposes only')
stringList = ''.join(map(str, charList))
print(stringList)

# Displays interactive model flow in tensorboard
def import_to_tensorboard(model_dir, log_dir, tag_set):
    with session.Session(graph=ops.Graph()) as sess:
        input_graph_def = saved_model_utils.get_meta_graph_def(model_dir,
                                                               tag_set).graph_def
        importer.import_graph_def(input_graph_def)

        pb_visual_writer = summary.FileWriter(log_dir)
        pb_visual_writer.add_graph(sess.graph)
        print("Model Imported. Visualize by running: "
              "tensorboard --logdir={}".format(log_dir))


import_to_tensorboard(model_dir, log_dir, tag_set)
