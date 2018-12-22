__author__ = 'yunbo'

import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTMCell as cslstm

def rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1,
        seq_length=20, input_length=10, tln=True):

    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None

    for t in range(seq_length-1):
        reuse = bool(gen_images)
        with tf.variable_scope('predrnn_pp', reuse=reuse):
            if t < input_length:
                inputs = images[:,t]
            else:
                inputs = mask_true[:,t-input_length]*images[:,t] + (1-mask_true[:,t-input_length])*x_gen

            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)

            x_gen = tf.layers.conv2d(inputs=hidden[num_layers-1],
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name="back_to_pixel")
            gen_images.append(x_gen)

    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    loss = tf.nn.l2_loss(gen_images - images[:,1:])
    loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    return [gen_images, loss]


def rnn_inference(images, num_layers, num_hidden, filter_size, stride=1,
        pred_length=11, tln=True):

    last_gen_image = None
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]
    input_length = tf.shape(images)[1]

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)

        with tf.variable_scope('states_layer%d' % i, reuse=False) as scope:
            try:
                c = tf.get_variable('c')
                cell.append(c)
            except ValueError:
                cell.append(None)
            try:
                h = tf.get_variable('h')
                hidden.append(h)
            except ValueError:
                hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    with tf.variable_scope('states_global', reuse=False) as scope:
        try:
            men = tf.get_variable('men')
        except ValueError:
            men = None
        try:
            z_t = tf.get_variable('z_t')
        except ValueError:
            z_t = None

    t = 0
    while t + 1 - pred_length < input_length:
        with tf.variable_scope('predrnn_pp', reuse=tf.AUTO_REUSE):
            if input_length > t:
                inputs = images[:,t]
            else:
                inputs = x_gen

            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)

            x_gen = tf.layers.conv2d(inputs=hidden[num_layers-1],
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name="back_to_pixel")
            last_gen_image = x_gen

        if t + 1 == input_length:
            for i in range(num_layers):
                with tf.variable_scope('states_layer%d' % i, reuse=tf.AUTO_REUSE) as scope:
                    h = tf.get_variable('h', hidden[i].get_shape())
                    h.assign(hidden[i])
                    c = tf.get_variable('c', cell[i].get_shape())
                    c.assign(cell[i])
                with tf.variable_scope('states_global', reuse=tf.AUTO_REUSE) as scope:
                    m = tf.get_variable('mem', mem.get_shape())
                    m.assign(mem)
                    z = tf.get_variable('z_t', z_t.get_shape())
                    z.assign(z_t)
        t += 1

    return last_gen_image

