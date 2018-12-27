from tensorflow.python.client import device_lib
import tensorflow as tf
import os


def available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [device.name for device in local_device_protos if device.device_type == 'GPU']


class Logger(object):
    Scalar = tf.summary.scalar
    Image  = tf.summary.image
    Histo  = tf.summary.histogram
    
    def __init__(self, folder, sess):
        self.sess = sess
        self.writer = tf.summary.FileWriter(folder, sess.graph)
        self.summaries = {}
        self.holders = {}

    def define_item(self, name, category, shape, dtype=tf.float32):
        self.holders[name] = tf.placeholder(dtype, shape=shape, name=name)
        self.summaries[name] = category(name, self.holders[name])

    def add(self, name, value, step):
        s = self.sess.run(self.summaries[name], feed_dict={self.holders[name]: value})
        self.writer.add_summary(s, step)


def smooth_L1(x):
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r
