__author__ = 'yunbo'
# modified by filick

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
tf.app.flags.DEFINE_string('dataset_name', 'mnist',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/moving-mnist-example/moving-mnist-train.npz',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/mnist_predrnn_pp',
                           'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'results/mnist_predrnn_pp',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 10,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 20,
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
# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 80000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 2000,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                            'number of iters saving models.')

assert FLAGS.input_length + 1 == FLAGS.seq_length


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_sum(grad, 0) / FLAGS.batch_size

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class Model(object):
    def __init__(self):
        self.gpus = tf_util.available_gpus()
        self.num_gpus = len(self.gpus)
        if self.num_gpus:
            assert FLAGS.batch_size % self.num_gpus == 0, "Batch size should be an integral multiple of number of GPUs"
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 FLAGS.seq_length,
                                 FLAGS.img_width // FLAGS.patch_size,
                                 FLAGS.img_width // FLAGS.patch_size,
                                 FLAGS.patch_size * FLAGS.patch_size * FLAGS.img_channel])

        x_splits = tf.split(self.x, max(self.num_gpus, 1))

        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        print(num_hidden)
        num_layers = len(num_hidden)

        opt = tf.train.AdamOptimizer(FLAGS.lr)
        # opt = tf.train.GradientDescentOptimizer(FLAGS.lr)

        pred_seq = []
        tower_grads = []
        tower_losses = []
        devices = self.gpus or ['/cpu:0']
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            for i, d in enumerate(devices):
                with tf.device(d), tf.name_scope('tower_%d' % i):
                    output_list = models_factory.construct_model(
                        FLAGS.model_name,
                        x_splits[i], None,
                        num_layers, num_hidden,
                        FLAGS.filter_size, FLAGS.stride,
                        FLAGS.seq_length, FLAGS.input_length,
                        FLAGS.layer_norm)
                    gen_ims = output_list[0]
                    loss = output_list[1]
                    pred_ims = gen_ims
                    # self.loss_train = loss / FLAGS.batch_size
                    # gradients
                    with tf.name_scope("compute_gradients"):
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

                    tower_losses.append(loss)
                    pred_seq.append(pred_ims)
                    outer_scope.reuse_variables()

        with tf.name_scope("apply_gradients"), tf.device(devices[0]):
            self.loss_train = tf.add_n(tower_losses) / FLAGS.batch_size
            mean_grads = average_gradients(tower_grads)
            global_step = tf.train.get_or_create_global_step()
            self.train_op = opt.apply_gradients(mean_grads, global_step)
            self.pred_seq = tf.concat(pred_seq, 0)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs):
        feed_dict = {self.x: inputs}
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)
    tf.gfile.MakeDirs(FLAGS.save_dir)
    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size, FLAGS.img_width, FLAGS.seq_length)

    print("Initializing models")
    model = Model()
    lr = FLAGS.lr

    # Prepare tensorboard logging
    logger = Logger(os.path.join(FLAGS.gen_frm_dir, 'board'), model.sess)
    logger.define_item("loss", Logger.Scalar, ())
    logger.define_item("lr", Logger.Scalar, ())
    logger.define_item("mse", Logger.Scalar, ())
    logger.define_item("psnr", Logger.Scalar, ())
    logger.define_item("fmae", Logger.Scalar, ())
    logger.define_item("ssim", Logger.Scalar, ())
    logger.define_item("sharp", Logger.Scalar, ())
    logger.define_item("image", Logger.Image, (1, 2 * FLAGS.img_width, FLAGS.img_width, FLAGS.img_channel), dtype='uint8')

    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, FLAGS.patch_size)

        logger.add('lr', lr, itr)
        cost = model.train(ims, lr)
        if FLAGS.reverse_input:
            ims_rev = ims[:, ::-1]
            cost += model.train(ims_rev, lr, mask_true)
            cost = cost / 2
        logger.add('loss', cost, itr)

        if itr % FLAGS.display_interval == 0:
            print('itr: ' + str(itr))
            print('training loss: ' + str(cost))

        if itr % FLAGS.test_interval == 0:
            print('test...')
            test_input_handle.begin(do_shuffle=False)
            res_path = os.path.join(FLAGS.gen_frm_dir, str(itr))
            os.mkdir(res_path)
            avg_mse = 0
            batch_id = 0
            img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                img_mse.append(0)
                ssim.append(0)
                psnr.append(0)
                fmae.append(0)
                sharp.append(0)
            while(test_input_handle.no_batch_left() == False):
                batch_id = batch_id + 1
                test_ims = test_input_handle.get_batch()
                test_dat = preprocess.reshape_patch(test_ims, FLAGS.patch_size)
                img_gen = model.test(test_dat)

                # concat outputs of different gpus along batch
                # img_gen = np.concatenate(img_gen)
                img_gen = preprocess.reshape_patch_back(
                    img_gen[:, np.newaxis, :, :, :], FLAGS.patch_size)
                # MSE per frame
                for i in range(1):
                    x = test_ims[:, -1, :, :, 0]
                    gx = img_gen[:, :, :, 0]
                    fmae[i] += metrics.batch_mae_frame_float(gx, x)
                    gx = np.maximum(gx, 0)
                    gx = np.minimum(gx, 1)
                    mse = np.square(x - gx).sum()
                    img_mse[i] += mse
                    avg_mse += mse

                    real_frm = np.uint8(x * 255)
                    pred_frm = np.uint8(gx * 255)
                    psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
                    for b in range(FLAGS.batch_size):
                        sharp[i] += np.max(
                            cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
                        score, _ = compare_ssim(
                            pred_frm[b], real_frm[b], full=True)
                        ssim[i] += score

                # save prediction examples
                if batch_id == 1:
                    sel = np.random.randint(FLAGS.batch_size)
                    img_seq_pd = img_gen[sel]
                    img_seq_gt = test_ims[sel, -1]
                    h, w = img_gen.shape[1:3]
                    out_img = np.zeros(
                        (1, h * 2, w * 1, FLAGS.img_channel), dtype='uint8')
                    for i, img_seq in enumerate([img_seq_gt, img_seq_pd]):
                        img = img_seq
                        img = np.maximum(img, 0)
                        img = np.uint8(img * 10)
                        img = np.minimum(img, 255)
                        out_img[0, (i * h):(i * h + h), :] = img
                    logger.add("image", out_img, itr)

                test_input_handle.next()
            avg_mse = avg_mse / (batch_id * FLAGS.batch_size)
            logger.add('mse', avg_mse, itr)
            print('mse per seq: ' + str(avg_mse))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(img_mse[i] / (batch_id * FLAGS.batch_size))
            psnr = np.asarray(psnr, dtype=np.float32) / batch_id
            fmae = np.asarray(fmae, dtype=np.float32) / batch_id
            ssim = np.asarray(ssim, dtype=np.float32) / \
                (FLAGS.batch_size * batch_id)
            sharp = np.asarray(sharp, dtype=np.float32) / \
                (FLAGS.batch_size * batch_id)
            print('psnr per frame: ' + str(np.mean(psnr)))
            logger.add('psnr', np.mean(psnr), itr)
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(psnr[i])
            print('fmae per frame: ' + str(np.mean(fmae)))
            logger.add('fmae', np.mean(fmae), itr)
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(fmae[i])
            print('ssim per frame: ' + str(np.mean(ssim)))
            logger.add('ssim', np.mean(ssim), itr)
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(ssim[i])
            print('sharpness per frame: ' + str(np.mean(sharp)))
            logger.add('sharp', np.mean(sharp), itr)
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(sharp[i])

        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)

        train_input_handle.next()


if __name__ == '__main__':
    tf.app.run()
