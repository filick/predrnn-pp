import time
import numpy as np
import tensorflow as tf
import cv2
import sys
import random
import os
from nets import models_factory
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics
from utils import tf_util
from utils.tf_util import Logger
from skimage.measure import compare_ssim

# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('input_images_path', '',
                           'path of input sequence, seperated with comma')
tf.app.flags.DEFINE_string('save_name', '',
                           'path and file name to save the result')

# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp_inference',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('pred_length', 11,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('img_width', 64,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_channel', 1,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_string('num_hidden', '128,64,64,64',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size', 4,
                            'patch size on one dimension.')
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# inference
tf.app.flags.DEFINE_integer('batch_size', 1,
                            'batch size for inference.')
tf.app.flags.DEFINE_boolean('new_states', False,
                            'whether to run model based on new states')


class Model(object):
    def __init__(self):
        self.gpus = tf_util.available_gpus()
        self.num_gpus = len(self.gpus)
        if self.num_gpus:
            assert FLAGS.batch_size % self.num_gpus == 0, "Batch size should be an integral multiple of number of GPUs"
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 None,
                                 FLAGS.img_width // FLAGS.patch_size,
                                 FLAGS.img_width // FLAGS.patch_size,
                                 FLAGS.patch_size * FLAGS.patch_size * FLAGS.img_channel])

        x_splits = tf.split(self.x, max(self.num_gpus, 1))

        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        num_layers = len(num_hidden)

        pred_seq = []
        devices = self.gpus or ['/cpu:0']
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            for i, d in enumerate(devices):
                with tf.device(d), tf.name_scope('tower_%d' % i):
                    pred_ims = models_factory.construct_model(
                        FLAGS.model_name, x_splits[i], None,
                        num_layers, num_hidden,
                        FLAGS.filter_size, FLAGS.stride,
                        FLAGS.pred_length, None,
                        FLAGS.layer_norm)
                    pred_seq.append(pred_ims)
                    outer_scope.reuse_variables()

        with tf.name_scope("apply_gradients"), tf.device(devices[0]):
            self.pred_seq = tf.concat(pred_seq, 0)

        # session
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def inference(self, inputs):
        feed_dict = {self.x: inputs}
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims


def main(argv=None):

    print("Initializing models")
    model = Model()


if __name__ == '__main__':
    tf.app.run()
