import tensorflow as tf
from nets import models_factory
from utils import preprocess
import numpy as np


class Model(object):
    def __init__(self):
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [1, 20, 16, 16, 16])

        self.mask_true = tf.placeholder(tf.float32,
                                        [1, 9, 16, 16, 16])

        loss_train = []
        self.pred_seq = []
        self.tf_lr = 0.001
        num_hidden = [128, 64, 64, 64]
        num_layers = len(num_hidden)
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            output_list = models_factory.construct_model(
                'predrnn_pp', self.x,
                self.mask_true,
                num_layers, num_hidden,
                5, 1, 20, 10, True)
            gen_ims = output_list[0]
            loss = output_list[1]
            pred_ims = gen_ims[:,9:]
            self.loss_train = loss / 1
            # gradients
            self.pred_seq.append(pred_ims)

        self.train_op = tf.train.AdamOptimizer(self.tf_lr).minimize(loss)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)

    def train(self, inputs, lr, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        loss, _, gen_ims = self.sess.run((self.loss_train, self.train_op, self.pred_seq), feed_dict)
        return loss, gen_ims


def main(argv=None):

    print("Initializing models")
    model = Model()
    lr = 0.001

    delta = 0.00002
    base = 0.99998
    eta = 1

    for itr in range(1, 21):
        ims = np.random.rand(1, 20, 64, 64, 1)
        ims = preprocess.reshape_patch(ims, 4)

        if itr < 50000:
            eta -= delta
        else:
            eta = 0.0
        random_flip = np.random.random_sample((1,9))
        true_token = (random_flip < eta)
        #true_token = (random_flip < pow(base,itr))
        mask_true = np.zeros([1,9,16,16,16], 'float32')
        for i in range(1):
            for j in range(9):
                if true_token[i,j]:
                    mask_true[i, j, :] = 1
                else:
                    mask_true[i, j, :] = 0
        cost, img_gen = model.train(ims, lr, mask_true)

        print('itr: ' + str(itr))
        print('training loss: ' + str(cost))

if __name__ == '__main__':
    main()
