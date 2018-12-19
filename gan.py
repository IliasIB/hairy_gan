import os
from random import randint, choice

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.nn import conv2d
from skimage import transform, io
from sklearn.utils import shuffle


def encoder(image):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
        # Input Layer
        input_layer = tf.reshape(image, [-1, 128, 128, 3])

        # Convolution Layer 1
        w_conv1 = tf.get_variable('e_wconv1', [6, 6, 3, 64])
        b_conv1 = tf.get_variable('e_bconv1', [64])
        h_conv1 = tf.nn.relu(conv2d(input_layer, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

        # Pooling Layer 1
        h_pool_1 = tf.layers.max_pooling2d(inputs=h_conv1, pool_size=[2, 2], strides=2)

        # Convolution Layer 2
        w_conv2 = tf.get_variable('e_wconv2', [4, 4, 64, 128])
        b_conv2 = tf.get_variable('e_bconv2', [128])
        h_conv2 = tf.nn.relu(conv2d(h_pool_1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

        # Pooling Layer 2
        h_pool_2 = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2)

        # Convolution Layer 3
        w_conv3 = tf.get_variable('e_wconv3', [4, 4, 128, 256])
        b_conv3 = tf.get_variable('e_bconv3', [256])
        h_conv3 = tf.nn.relu(conv2d(h_pool_2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

        # Pooling Layer 3
        h_pool_3 = tf.layers.max_pooling2d(inputs=h_conv3, pool_size=[2, 2], strides=2)

        # Convolution Layer 4
        w_conv4 = tf.get_variable('e_wconv4', [4, 4, 256, 256])
        b_conv4 = tf.get_variable('e_bconv4', [256])
        h_conv4 = tf.nn.relu(conv2d(h_pool_3, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

        # Pooling Layer 4
        h_pool_4 = tf.layers.max_pooling2d(inputs=h_conv4, pool_size=[2, 2], strides=2)

        # Fully connected layer 1
        w_fc_1 = tf.get_variable('e_wfc1', [8 * 8 * 256, 256])
        b_fc_1 = tf.get_variable('e_bfc1', [256])
        h_pool4_flat_1 = tf.reshape(h_pool_4, [-1, 8 * 8 * 256])
        average = tf.nn.relu(tf.matmul(h_pool4_flat_1, w_fc_1) + b_fc_1)

        # Fully connected layer 2
        w_fc_2 = tf.get_variable('e_wfc2', [8 * 8 * 256, 256])
        b_fc_2 = tf.get_variable('e_bfc2', [256])
        h_pool4_flat_2 = tf.reshape(h_pool_4, [-1, 8 * 8 * 256])
        deviation = tf.nn.relu(tf.matmul(h_pool4_flat_2, w_fc_2) + b_fc_2)

    return average, deviation


def generator(z, sample_dimension, y, batch_size):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        # Concatenate hair style parameter
        z = tf.concat([z, y], 1)

        # Fully connected layer 1
        w_fc_1 = tf.get_variable('g_wfc1', [sample_dimension + 64, 8064])
        b_fc_1 = tf.get_variable('g_bfc1', [8064])
        h_fc_1 = tf.nn.relu(tf.matmul(z, w_fc_1) + b_fc_1)

        # Fully connected layer 2
        w_fc_2 = tf.get_variable('g_wfc2', [8064, 4 * 4 * 504])
        b_fc_2 = tf.get_variable('g_bfc2', [4 * 4 * 504])
        h_fc_2 = tf.nn.relu(tf.matmul(h_fc_1, w_fc_2) + b_fc_2)

        # Hidden Layer
        hidden_layer = tf.reshape(h_fc_2, [-1, 4, 4, 504])
        hidden_layer = tf.nn.relu(hidden_layer)

        # DeConv Layer 1
        w_conv1 = tf.get_variable('g_wconv1', [3, 3, 288, 504])
        b_conv1 = tf.get_variable('g_bconv1', [288])
        h_conv1 = tf.nn.conv2d_transpose(hidden_layer, w_conv1, output_shape=[batch_size, 8, 8, 288],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        h_conv1 = tf.contrib.layers.batch_norm(inputs=h_conv1, scale=True, scope="g_bn1")
        h_conv1 = tf.nn.relu(h_conv1)

        # DeConv Layer 2
        w_conv2 = tf.get_variable('g_wconv2', [3, 3, 216, 288])
        b_conv2 = tf.get_variable('g_bconv2', [216])
        h_conv2 = tf.nn.conv2d_transpose(h_conv1, w_conv2, output_shape=[batch_size, 16, 16, 216],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv2
        h_conv2 = tf.contrib.layers.batch_norm(inputs=h_conv2, scale=True, scope="g_bn2")
        h_conv2 = tf.nn.relu(h_conv2)

        # DeConv Layer 3
        w_conv3 = tf.get_variable('g_wconv3', [5, 5, 144, 216])
        b_conv3 = tf.get_variable('g_bconv3', [144])
        h_conv3 = tf.nn.conv2d_transpose(h_conv2, w_conv3, output_shape=[batch_size, 32, 32, 144],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv3
        h_conv3 = tf.contrib.layers.batch_norm(inputs=h_conv3, scale=True, scope="g_bn3")
        h_conv3 = tf.nn.relu(h_conv3)

        # DeConv Layer 4
        w_conv4 = tf.get_variable('g_wconv4', [5, 5, 72, 144])
        b_conv4 = tf.get_variable('g_bconv4', [72])
        h_conv4 = tf.nn.conv2d_transpose(h_conv3, w_conv4, output_shape=[batch_size, 64, 64, 72],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv4
        h_conv4 = tf.contrib.layers.batch_norm(inputs=h_conv4, scale=True, scope="g_bn4")
        h_conv4 = tf.nn.relu(h_conv4)

        # DeConv Layer 5
        w_conv5 = tf.get_variable('g_wconv5', [6, 6, 3, 72])
        b_conv5 = tf.get_variable('g_bconv5', [3])
        h_conv5 = tf.nn.conv2d_transpose(h_conv4, w_conv5, output_shape=[batch_size, 128, 128, 3],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv5
        h_conv5 = tf.nn.tanh(h_conv5)
    return h_conv5


def discrimator(image):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        # Convolution Layer 1
        w_conv1 = tf.get_variable('d_wconv1', [6, 6, 3, 64])
        b_conv1 = tf.get_variable('d_bconv1', [64])
        h_conv1 = tf.nn.relu(conv2d(image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

        # Pooling Layer 1
        h_pool_1 = tf.layers.max_pooling2d(inputs=h_conv1, pool_size=[2, 2], strides=2)

        # Convolution Layer 2
        w_conv2 = tf.get_variable('d_wconv2', [4, 4, 64, 128])
        b_conv2 = tf.get_variable('d_bconv2', [128])
        h_conv2 = tf.nn.relu(conv2d(h_pool_1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

        # Pooling Layer 2
        h_pool_2 = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2)

        # Convolution Layer 3
        w_conv3 = tf.get_variable('d_wconv3', [4, 4, 128, 128])
        b_conv3 = tf.get_variable('d_bconv3', [128])
        h_conv3 = tf.nn.relu(conv2d(h_pool_2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

        # Pooling Layer 3
        h_pool_3 = tf.layers.max_pooling2d(inputs=h_conv3, pool_size=[2, 2], strides=2)

        # Convolution Layer 4
        w_conv4 = tf.get_variable('d_wconv4', [4, 4, 128, 256])
        b_conv4 = tf.get_variable('d_bconv4', [256])
        h_conv4 = tf.nn.relu(conv2d(h_pool_3, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

        # Pooling Layer 4
        h_pool_4 = tf.layers.max_pooling2d(inputs=h_conv4, pool_size=[2, 2], strides=2)

        # Fully connected layer
        w_fc = tf.get_variable('d_wfc1', [8 * 8 * 256, 1])
        b_fc = tf.get_variable('d_bfc1', [1])
        h_pool4_flat = tf.reshape(h_pool_4, [-1, 8 * 8 * 256])
        h_fc = tf.matmul(h_pool4_flat, w_fc) + b_fc
    return h_fc, tf.nn.sigmoid(h_fc), h_pool_4


def recognizer(discriminator, y_size):
    # Fully connected layer for y
    w_fc = tf.get_variable('r_wfc1', [8 * 8 * 256, y_size])
    b_fc = tf.get_variable('r_bfc1', [y_size])
    h_pool4_flat = tf.reshape(discriminator, [-1, 8 * 8 * 256])
    q_y_given_x = tf.nn.softmax(tf.matmul(h_pool4_flat, w_fc) + b_fc)

    # Fully connected layer for z
    w_fc_2 = tf.get_variable('r_wfc2', [8 * 8 * 256, 256])
    b_fc_2 = tf.get_variable('r_bfc2', [256])
    h_pool4_flat_2 = tf.reshape(discriminator, [-1, 8 * 8 * 256])
    q_z_given_x = tf.nn.softmax(tf.matmul(h_pool4_flat, w_fc) + b_fc)

    return q_y_given_x, q_z_given_x
recognizer

def train_reconstruction(batch_size, epoch_amount):
    image = tf.placeholder('float32', [None, 128, 128, 3], name="reconstruction_training_image_input")
    y_input = tf.placeholder('float32', [None, 64], name="reconstruction_training_y_input")
    z_rand_input = tf.placeholder('float32', [None, 256], name="reconstruction_training_z_rand_input")
    y_rand_input = tf.placeholder('float32', [None, 64], name="reconstruction_training_y_rand_input")
    average, deviation = encoder(image)

    # Generate sample
    sample = np.random.uniform(-1, 1, [batch_size, 256])
    z = average + sample * deviation
    decoder_f = generator(z, 256, y_input, batch_size)

    # Encoder loss
    enc_loss = encoder_loss(average, deviation, decoder_f, image, batch_size)

    # Standard generator loss
    fake_logits, fake, discrimator_f = discrimator(decoder_f)
    # Calculates the probabilities of fakes labeled as true by the discriminator
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_logits, labels=tf.ones_like(fake_logits)
    )

    

    # Recognition loss for z
    # Term 1: For image give z
    _, _, discriminator_f_real = discriminator(image)
    _, q_z_given_x_real = recognizer(discrimator_f_real, 64)
    loss_rec_z_1 = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=q_z_given_x_real, labels=tf.ones_like(z)
    )

    # Term 2: For image give reconstructed z 
    decoder_f = generator(z, 256, y_input, batch_size)
    _, _, discriminator_f = discriminator(decoder_f)
    _, q_z_given_x = recognizer(discriminator_f, 64)
    loss_rec_z_2 = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=q_z_given_x, labels=tf.ones_like(z)
    )

    # Term 3: For generated image give reconstructed z
    decoder_f_rand = generator(z_rand_input, 256, y_input, batch_size)
    _, _, discriminator_f_rand = discriminator(decoder_f_rand)
    _, q_z_given_rand_x = recognizer(discrimator_f_rand, 64)
    loss_rec_z_3 = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=q_z_given_rand_x, labels=tf.ones_like(z_rand_input)
    )


    # Term 4: For image, change hair style and give reconstructed z 
    decoder_f_mod = generator(z, 256, y_rand_input, batch_size)
    _, _, discriminator_f_mod = discriminator(decoder_f_mod)
    _, q_z_given_mod_x = recognizer(discriminator_f_mod, 64)
    loss_rec_z_4 = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=q_z_given_mod_x, labels=tf.ones_like(z)
    )

    loss_recognition_z = -loss_rec_z_1 - loss_rec_z_2 - loss_rec_z_3 - loss_rec_z_4


    # Recognition loss for y
    # Term 1: For image, change hair style and give reconstructed hair style
    decoder_f_mod = generator(z, 256, y_rand_input, batch_size)
    _, _, discriminator_f_mod = discriminator(decoder_f_mod)
    q_y_given_mod_x, _ = recognizer(discriminator_f_mod, 64)
    loss_rec_y_1 = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=q_y_given_mod_x, labels=tf.ones_like(y_rand_input)
    )

    # Term 2: For image give reconstructed hair style
    q_y_given_g, _ = recognizer(discrimator_f, 64)
    loss_rec_y_2 = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=q_y_given_g, labels=tf.ones_like(y_input)
    )

    # Term 3: For generated image give reconstructed hair style
    decoder_f_rand = generator(z_rand_input, 256, y_input, batch_size)
    _, _, discriminator_f_rand = discriminator(decoder_f_rand)
    q_y_given_rand_x, _ = recognizer(discrimator_f_rand, 64)
    loss_rec_y_3 = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=q_y_given_rand_x, labels=tf.ones_like(y_rand_input)
    )

    loss_recognition_y = -loss_rec_y_1 - loss_rec_y_2 - loss_rec_y_3

    # cross_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(q_y_given_g + 1e-8) * y_input, 1))
    # ent = tf.reduce_mean(-tf.reduce_sum(tf.log(y_input + 1e-8) * y_input, 1))
    # q_loss = cross_ent + ent
    q_loss = loss_recognition_z + loss_recognition_y

    # Reconstruction loss for decoder
    loss_rec_decoder = tf.losses.mean_squared_error(labels=image, predictions=decoder_f)

    # Total generator loss
    g_loss = g_loss + q_loss + loss_rec_decoder

    # Variable list
    tvars = tf.trainable_variables()
    enc_vars = [var for var in tvars if 'e_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]
    q_vars = [var for var in tvars if 'r_' in var.name]

    # Solvers
    g_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars + q_vars)
    e_solver = tf.train.AdamOptimizer().minimize(enc_loss, var_list=enc_vars + g_vars)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto(
        # device_count={'GPU': 0}
    )

    with tf.Session(config=config) as session:
        # Run the initializer
        session.run(init)

        var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
                     for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        print(sum(var_sizes) / (1024 ** 2), 'MB')

        for epoch in range(1, epoch_amount):
            print('Epoch:', epoch)

            x_training, y_training = get_training_set(batch_size)
            x_training, y_training = shuffle(x_training, y_training)

            # Image iterator
            # image_dataset = tf.data.Dataset.from_tensor_slices(x_training).batch(batch_size)
            # iterator_data = image_dataset.make_initializable_iterator()
            # image_iterator = iterator_data.get_next()

            # Label iterator
            # label_dataset = tf.data.Dataset.from_tensor_slices(y_training).batch(batch_size)
            # iterator_labels = label_dataset.make_initializable_iterator()
            # label_iterator = iterator_labels.get_next()
            #
            # session.run(iterator_data.initializer)
            # session.run(iterator_labels.initializer)

            # for i in range(1, int(x_training.shape[0] / batch_size) + 1):
            # image_batch = session.run(image_iterator)
            # label_batch = session.run(label_iterator)
            # _, g_loss_curr = session.run([g_solver, g_loss],
            #                              feed_dict={image: image_batch, y_input: label_batch})
            _, e_loss_curr = session.run([e_solver, enc_loss],
                                         feed_dict={image: x_training, y_input: y_training})
            print('Encoder loss:', e_loss_curr)
        inputs = {
                "image_placeholder": image,
                "y_input_placeholder": y_input,
        }
        outputs = {"prediction": q_y_given_g}
        tf.saved_model.simple_save(session, 'weights/encoder.ckpt', inputs, outputs)
        test(session)


def encoder_loss(average, deviation, decoder_f, image, batch_size):
    # encode_decode_loss = images * tf.log(epsilon + decoder_f) + (1 - images) * (tf.log(epsilon + 1 - decoder_f))
    decoder_f = tf.reshape(decoder_f, [batch_size, 128*128*3])
    image = tf.reshape(image, [batch_size, 128*128*3])
    encode_decode_loss = tf.losses.mean_squared_error(labels=image, predictions=decoder_f)
    # encode_decode_loss = tf.reduce_sum(encode_decode_loss, 1)

    # Latent space regularization
    kl_div_loss = 1 + deviation - tf.square(average) - tf.exp(deviation)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    # Encoder loss
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)


def train_modification():
    pass


def train_generation():
    pass


def test(session):
    img = io.imread('Tenkind/6/1D-Bald-2.jpg')
    resized = transform.resize(img, (128, 128, 3))
    resized = np.reshape(resized, (1, 128, 128, 3))
    y = np.zeros(64, dtype=float)
    y[0] = 1
    y = np.reshape(y, (1, 64))

    # plt.imshow(resized)
    # plt.show()

    # Inputs
    image_input = tf.placeholder('float32', [None, 128, 128, 3], name="test_image_input")
    y_input = tf.placeholder('float32', [None, 64], name="test_y_input")

    # Encode image
    average, deviation = encoder(image_input)

    # Generate sample
    sample = np.random.uniform(-1, 1, [1, 256])
    z = average + sample * deviation

    # Generate image
    sample_image = generator(z, 256, y_input, 1)

    generated_image = (session.run(sample_image, feed_dict={image_input: resized, y_input: y}))

    generated_image = np.reshape(generated_image, (128, 128, 3))
    plt.imshow(generated_image)
    plt.show()


def get_training_set(load_amount):
    x_train = []
    y_train = []
    amount_loaded = 1
    while amount_loaded <= load_amount:
        random_folder = randint(0, 63)
        folder = "/home/ilias/Repositories/hairy_gan/60kind/" + str(random_folder)
        files = os.listdir(folder)
        try:
            image = io.imread(os.path.join(folder, choice(files)))
            resized = transform.resize(image, (128, 128, 3))
            if resized.shape == (128, 128, 3):
                x_train.append(resized)
                iter_y = np.zeros(64)
                np.put(iter_y, [random_folder], [1])
                y_train.append(iter_y)
                amount_loaded += 1
        except ValueError:
            pass
        except IOError:
            pass

    return np.array(x_train), np.array(y_train)


if __name__ == "__main__":
    train_reconstruction(10, 15000)
