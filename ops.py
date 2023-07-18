from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import tensorflow as tf


def d_loss_fn(f_logit, r_logit):
    f_loss = tf.reduce_mean(f_logit)
    r_loss = tf.reduce_mean(r_logit)
    return f_loss - r_loss


def wasserstein_loss(f_logit):
    f_loss = -tf.reduce_mean(f_logit)
    return f_loss
