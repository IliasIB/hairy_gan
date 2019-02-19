import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.nn import conv2d
from skimage import transform, io
from sklearn.utils import shuffle
from tensorflow.python.saved_model import tag_constants


def encoder(image):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as _:
        # Input Layer
        input_layer = tf.reshape(image, [-1, 128, 128, 3], 'e_input')

        # Convolution Layer 1
        w_conv1 = tf.get_variable('e_wconv1', [6, 6, 3, 64])
        b_conv1 = tf.get_variable('e_bconv1', [64])
        h_conv1 = tf.nn.relu(conv2d(input_layer, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, 'e_relu1')

        # Pooling Layer 1
        h_pool_1 = tf.layers.max_pooling2d(inputs=h_conv1, pool_size=[2, 2], strides=2, name='e_pool1')

        # Convolution Layer 2
        w_conv2 = tf.get_variable('e_wconv2', [4, 4, 64, 128])
        b_conv2 = tf.get_variable('e_bconv2', [128])
        h_conv2 = tf.nn.relu(conv2d(h_pool_1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, name='e_relu2')

        # Pooling Layer 2
        h_pool_2 = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2, name='e_pool2')

        # Convolution Layer 3
        w_conv3 = tf.get_variable('e_wconv3', [4, 4, 128, 256])
        b_conv3 = tf.get_variable('e_bconv3', [256])
        h_conv3 = tf.nn.relu(conv2d(h_pool_2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3, name='e_relu3')

        # Pooling Layer 3
        h_pool_3 = tf.layers.max_pooling2d(inputs=h_conv3, pool_size=[2, 2], strides=2, name='e_pool3')

        # Convolution Layer 4
        w_conv4 = tf.get_variable('e_wconv4', [4, 4, 256, 256])
        b_conv4 = tf.get_variable('e_bconv4', [256])
        h_conv4 = tf.nn.relu(conv2d(h_pool_3, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4, name='e_relu4')

        # Pooling Layer 4
        h_pool_4 = tf.layers.max_pooling2d(inputs=h_conv4, pool_size=[2, 2], strides=2, name='e_pool4')

        # Fully connected layer 1
        w_fc_1 = tf.get_variable('e_wfc1', [8 * 8 * 256, 256])
        b_fc_1 = tf.get_variable('e_bfc1', [256])
        h_pool4_flat_1 = tf.reshape(h_pool_4, [-1, 8 * 8 * 256], name='e_rfc1')
        average = tf.nn.relu(tf.matmul(h_pool4_flat_1, w_fc_1) + b_fc_1, name='e_average')

        # Fully connected layer 2
        w_fc_2 = tf.get_variable('e_wfc2', [8 * 8 * 256, 256])
        b_fc_2 = tf.get_variable('e_bfc2', [256])
        h_pool4_flat_2 = tf.reshape(h_pool_4, [-1, 8 * 8 * 256], name='e_rfc2')
        deviation = tf.nn.relu(tf.matmul(h_pool4_flat_2, w_fc_2) + b_fc_2, name='e_deviation')

    return average, deviation


def generator(z, sample_dimension, y, batch_size):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as _:
        # Concatenate hair style parameter
        z = tf.concat([z, y], 1)

        # Fully connected layer 1
        w_fc_1 = tf.get_variable('g_wfc1', [sample_dimension + 10, 8064])
        b_fc_1 = tf.get_variable('g_bfc1', [8064])
        h_fc_1 = tf.nn.relu(tf.matmul(z, w_fc_1) + b_fc_1, name='g_relufc1')

        # Fully connected layer 2
        w_fc_2 = tf.get_variable('g_wfc2', [8064, 4 * 4 * 504])
        b_fc_2 = tf.get_variable('g_bfc2', [4 * 4 * 504])
        h_fc_2 = tf.nn.relu(tf.matmul(h_fc_1, w_fc_2) + b_fc_2, name='g_relufc2')

        # Hidden Layer
        hidden_layer = tf.reshape(h_fc_2, [-1, 4, 4, 504], name='g_rh')
        hidden_layer = tf.nn.relu(hidden_layer, name='g_reluh')

        # DeConv Layer 1
        w_conv1 = tf.get_variable('g_wconv1', [3, 3, 288, 504])
        b_conv1 = tf.get_variable('g_bconv1', [288])
        h_conv1 = tf.nn.conv2d_transpose(hidden_layer, w_conv1, output_shape=[batch_size, 8, 8, 288],
                                         strides=[1, 2, 2, 1], padding='SAME', name='g_tconv1') + b_conv1
        h_conv1 = tf.contrib.layers.batch_norm(inputs=h_conv1, scale=True, scope="g_bn1")
        h_conv1 = tf.nn.relu(h_conv1, name='g_relu1')

        # DeConv Layer 2
        w_conv2 = tf.get_variable('g_wconv2', [3, 3, 216, 288])
        b_conv2 = tf.get_variable('g_bconv2', [216])
        h_conv2 = tf.nn.conv2d_transpose(h_conv1, w_conv2, output_shape=[batch_size, 16, 16, 216],
                                         strides=[1, 2, 2, 1], padding='SAME', name='g_tconv2') + b_conv2
        h_conv2 = tf.contrib.layers.batch_norm(inputs=h_conv2, scale=True, scope="g_bn2")
        h_conv2 = tf.nn.relu(h_conv2, name='g_relu2')

        # DeConv Layer 3
        w_conv3 = tf.get_variable('g_wconv3', [5, 5, 144, 216])
        b_conv3 = tf.get_variable('g_bconv3', [144])
        h_conv3 = tf.nn.conv2d_transpose(h_conv2, w_conv3, output_shape=[batch_size, 32, 32, 144],
                                         strides=[1, 2, 2, 1], padding='SAME', name='g_tconv3') + b_conv3
        h_conv3 = tf.contrib.layers.batch_norm(inputs=h_conv3, scale=True, scope="g_bn3")
        h_conv3 = tf.nn.relu(h_conv3, name='g_relu3')

        # DeConv Layer 4
        w_conv4 = tf.get_variable('g_wconv4', [5, 5, 72, 144])
        b_conv4 = tf.get_variable('g_bconv4', [72])
        h_conv4 = tf.nn.conv2d_transpose(h_conv3, w_conv4, output_shape=[batch_size, 64, 64, 72],
                                         strides=[1, 2, 2, 1], padding='SAME', name='g_tconv4') + b_conv4
        h_conv4 = tf.contrib.layers.batch_norm(inputs=h_conv4, scale=True, scope="g_bn4")
        h_conv4 = tf.nn.relu(h_conv4, name='g_relu4')

        # DeConv Layer 5
        w_conv5 = tf.get_variable('g_wconv5', [6, 6, 3, 72])
        b_conv5 = tf.get_variable('g_bconv5', [3])
        h_conv5 = tf.nn.conv2d_transpose(h_conv4, w_conv5, output_shape=[batch_size, 128, 128, 3],
                                         strides=[1, 2, 2, 1], padding='SAME', name='g_tconv5') + b_conv5
        h_conv5 = tf.nn.tanh(h_conv5, 'g_tanh')
    return h_conv5


def discriminator(image):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as _:
        # Convolution Layer 1
        w_conv1 = tf.get_variable('d_wconv1', [6, 6, 3, 64])
        b_conv1 = tf.get_variable('d_bconv1', [64])
        h_conv1 = tf.nn.relu(conv2d(image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, name='d_rconv1')

        # Pooling Layer 1
        h_pool_1 = tf.layers.max_pooling2d(inputs=h_conv1, pool_size=[2, 2], strides=2, name='d_pool1')

        # Convolution Layer 2
        w_conv2 = tf.get_variable('d_wconv2', [4, 4, 64, 128])
        b_conv2 = tf.get_variable('d_bconv2', [128])
        h_conv2 = tf.nn.relu(conv2d(h_pool_1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, name='d_rconv2')

        # Pooling Layer 2
        h_pool_2 = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2, name='d_pool2')

        # Convolution Layer 3
        w_conv3 = tf.get_variable('d_wconv3', [4, 4, 128, 128])
        b_conv3 = tf.get_variable('d_bconv3', [128])
        h_conv3 = tf.nn.relu(conv2d(h_pool_2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3, name='d_rconv3')

        # Pooling Layer 3
        h_pool_3 = tf.layers.max_pooling2d(inputs=h_conv3, pool_size=[2, 2], strides=2, name='d_pool3')

        # Convolution Layer 4
        w_conv4 = tf.get_variable('d_wconv4', [4, 4, 128, 256])
        b_conv4 = tf.get_variable('d_bconv4', [256])
        h_conv4 = tf.nn.relu(conv2d(h_pool_3, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4, name='d_rconv4')

        # Pooling Layer 4
        h_pool_4 = tf.layers.max_pooling2d(inputs=h_conv4, pool_size=[2, 2], strides=2, name='d_pool4')

        # Fully connected layer
        w_fc = tf.get_variable('d_wfc1', [8 * 8 * 256, 1])
        b_fc = tf.get_variable('d_bfc1', [1])
        h_pool4_flat = tf.reshape(h_pool_4, [-1, 8 * 8 * 256], name='d_reshape1')
        h_fc = tf.matmul(h_pool4_flat, w_fc, name='d_mul') + b_fc
    return h_fc, tf.nn.sigmoid(h_fc, name='d_sigmoid'), h_pool_4


def recognizer(discriminator_f, y_size):
    with tf.variable_scope('recognizer', reuse=tf.AUTO_REUSE) as _:
        # Fully connected layer for y
        w_fc = tf.get_variable('r_wfc1', [8 * 8 * 256, y_size])
        b_fc = tf.get_variable('r_bfc1', [y_size])
        h_pool4_flat = tf.reshape(discriminator_f, [-1, 8 * 8 * 256], name='r_pool4_1')
        q_y_given_x = tf.nn.softmax(tf.matmul(h_pool4_flat, w_fc) + b_fc, name='r_q_y_given_x')

        # Fully connected layer for z
        w_fc_2 = tf.get_variable('r_wfc2', [8 * 8 * 256, 256])
        b_fc_2 = tf.get_variable('r_bfc2', [256])
        h_pool4_flat_2 = tf.reshape(discriminator_f, [-1, 8 * 8 * 256], name='r_pool4_2')
        q_z_given_x = tf.nn.softmax(tf.matmul(h_pool4_flat_2, w_fc_2) + b_fc_2, name='r_q_z_given_x')

    return q_y_given_x, q_z_given_x


def train_reconstruction(batch_size, iteration_amount, epoch_amount):
    # Prediction inputs
    image = tf.placeholder('float32', [None, 128, 128, 3], name="reconstruction_training_image_input")
    y_input = tf.placeholder('float32', [None, 10], name="reconstruction_training_y_input")
    z_rand_input = tf.placeholder('float32', [None, 256], name="reconstruction_training_z_rand_input")

    # Generate sample
    average, deviation = encoder(image)
    sample = np.random.uniform(-1, 1, [batch_size, 256])
    z = average + sample * deviation

    # Prediction outputs
    # Encoded
    decoder_encoded = generator(z, 256, y_input, batch_size)
    fake_logits, discriminator_encoded, discriminator_encoded_layer = discriminator(decoder_encoded)
    recognizer_encoded, _ = recognizer(discriminator_encoded_layer, 10)

    # Real
    _, discriminator_real, discriminator_real_layer = discriminator(image)
    recognizer_real, _ = recognizer(discriminator_real_layer, 10)

    # Random
    decoder_random = generator(z_rand_input, 256, y_input, batch_size)
    _, discriminator_random, discriminator_random_layer = discriminator(decoder_random)
    recognizer_random, _ = recognizer(discriminator_random_layer, 10)

    # Loss
    enc_loss = get_encoder_loss(decoder_encoded, image)
    kl_loss = get_kl_loss(average, deviation)
    g_loss = get_generator_loss(discriminator_encoded, discriminator_random,
                                recognizer_encoded, recognizer_random,
                                y_input)
    d_loss = get_discriminator_loss(discriminator_real, discriminator_encoded, discriminator_random)
    q_loss = get_recognizer_loss(recognizer_real, recognizer_encoded, recognizer_random,
                                 y_input)
    # gd_loss = get_discriminator_feature_matching_loss(discriminator_real_layer, discriminator_encoded_layer)
    # gq_loss = get_recognizer_feature_matching_loss(discriminator_real_layer, discriminator_encoded_layer)

    # Variable list
    tvars = tf.trainable_variables()
    enc_vars = [var for var in tvars if 'e_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]
    q_vars = [var for var in tvars if 'r_' in var.name]
    d_vars = [var for var in tvars if 'd_' in var.name]

    # Weights
    lambda_1 = 3
    lambda_2 = 1
    lambda_3 = 1e-3
    lambda_4 = 1e-3

    # Solvers
    e_solver = tf.train.AdamOptimizer(2e-4).minimize(enc_loss + lambda_1 * kl_loss, var_list=enc_vars)
    g_solver = tf.train.AdamOptimizer(2e-4).minimize(lambda_2*g_loss + enc_loss, var_list=g_vars)
    d_solver = tf.train.AdamOptimizer(2e-4).minimize(d_loss, var_list=d_vars)
    q_solver = tf.train.AdamOptimizer(2e-4).minimize(q_loss, var_list=q_vars)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto(
        # device_count={'GPU': 0}
    )

    with tf.Session(config=config) as session:
        # Run the initializer
        session.run(init)

        # Epochs
        for epoch in range(epoch_amount):
            for iteration in range(1, iteration_amount):
                # print('Iteration:', iteration)
                x_training, y_training = get_training_set(batch_size)
                x_training, y_training = shuffle(x_training, y_training)
                sample_z = np.random.uniform(-1, 1, [batch_size, 256])
                _, _, _, _ = session.run([e_solver, g_solver, d_solver, q_solver],
                                         feed_dict={image: x_training, y_input: y_training, z_rand_input: sample_z})
                # output_loss(d_loss, enc_loss, g_loss, image, kl_loss, q_loss, sample_z, session, x_training, y_input,
                #             y_training, z_rand_input)

            # single_test(session, decoder_encoded, image, y_input, batch_size)
            inputs = {
                "image_placeholder": image,
                "y_input_placeholder": y_input,
            }
            outputs = {"decoder": decoder_encoded}
            tf.saved_model.simple_save(session, 'weights/epoch-' + str(epoch), inputs, outputs)


def output_loss(d_loss, enc_loss, g_loss, image, kl_loss, q_loss, sample_z, session, x_training, y_input, y_training,
                z_rand_input):
    enc_loss_curr, g_loss_curr, d_loss_curr, q_loss_curr = \
        session.run([enc_loss + kl_loss, g_loss + enc_loss, d_loss, q_loss],
                    feed_dict={image: x_training, y_input: y_training, z_rand_input: sample_z})
    print("Encoder loss: ", enc_loss_curr)
    print("Generator loss: ", g_loss_curr)
    print("Discriminator loss: ", d_loss_curr)
    print("Recognizer loss: ", q_loss_curr)


def get_encoder_loss(decoder_encoded, image):
    # Encoder loss
    encode_decode_loss = 0.5 * tf.losses.mean_squared_error(labels=image, predictions=decoder_encoded)
    return encode_decode_loss


def get_kl_loss(average, deviation):
    # Latent space regularization
    kl_div_loss = 1 + deviation - tf.square(average) - tf.exp(deviation)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    return kl_div_loss


def get_generator_loss(discriminator_fake, discriminator_random, recognizer_encoded, recognizer_random, y_input):
    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_fake), discriminator_fake)
    g_loss += tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_random), discriminator_random)
    g_loss += tf.losses.softmax_cross_entropy(y_input, recognizer_encoded)
    g_loss += tf.losses.softmax_cross_entropy(y_input, recognizer_random)
    return g_loss


def get_discriminator_loss(discriminator_real, discriminator_fake, discriminator_random):
    d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_real), discriminator_real)
    d_loss += tf.losses.sigmoid_cross_entropy(tf.zeros_like(discriminator_fake), discriminator_fake)
    d_loss += tf.losses.sigmoid_cross_entropy(tf.zeros_like(discriminator_random), discriminator_random)
    return d_loss


def get_recognizer_loss(recognizer_real, recognizer_fake, recognizer_random, y_input):
    q_loss = tf.losses.softmax_cross_entropy(y_input, recognizer_real)
    q_loss += tf.losses.softmax_cross_entropy(y_input, recognizer_fake)
    q_loss += tf.losses.softmax_cross_entropy(y_input, recognizer_random)
    return q_loss


def get_discriminator_feature_matching_loss(discriminator_real_layer, discriminator_fake_layer):
    expected_real_features = tf.reduce_mean(discriminator_real_layer, axis=0)
    expected_fake_features = tf.reduce_mean(discriminator_fake_layer, axis=0)
    return 0.5 * tf.losses.mean_squared_error(expected_real_features, expected_fake_features)


def get_recognizer_feature_matching_loss(recognizer_real_layer, recognizer_fake_layer):
    expected_real_features = tf.reduce_mean(recognizer_real_layer, axis=0)
    expected_fake_features = tf.reduce_mean(recognizer_fake_layer, axis=0)
    return 0.5 * tf.losses.mean_squared_error(expected_real_features, expected_fake_features)


def single_test(session, decoder, image_input, y_input, batch_size):
    x_training, y_training = get_training_set(batch_size)
    generated_image = (session.run(decoder, feed_dict={image_input: x_training, y_input: y_training}))
    generated_image = (generated_image[0] + 1) / 2 * 255
    generated_image = np.uint8(generated_image)
    generated_image = np.clip(generated_image, 0, 255)
    plt.imshow(generated_image)
    plt.show()


def test(batch_size):
    restored_graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session() as session:
            tf.saved_model.loader.load(
                session,
                [tag_constants.SERVING],
                'weights/epoch-5/',
            )

            image_input = restored_graph.get_tensor_by_name('reconstruction_training_image_input:0')
            y_input = restored_graph.get_tensor_by_name('reconstruction_training_y_input:0')

            decoder = restored_graph.get_tensor_by_name('generator/g_tanh:0')

            for i in range(10):
                x_test, y_test = get_training_set(batch_size)

                generated_image = (session.run(decoder, feed_dict={image_input: x_test, y_input: y_test}))
                generated_image = (generated_image[0] + 1) / 2 * 255
                generated_image = np.uint8(generated_image)
                generated_image = np.clip(generated_image, 0, 255)
                x_test = (x_test[0] + 1) / 2 * 255
                x_test = np.uint8(x_test)
                x_test = np.clip(x_test, 0, 255)
                plt.imshow(x_test)
                plt.show()
                plt.imshow(generated_image)
                plt.show()


def get_training_set(load_amount):
    x_train = []
    y_train = []
    amount_loaded = 1
    while amount_loaded <= load_amount:
        random_folder = random.randint(0, 9)
        folder = "/home/ilias/Repositories/hairy_gan/Tenkind/" + str(random_folder)
        files = os.listdir(folder)
        try:
            image = io.imread(os.path.join(folder, random.choice(files)))
            resized = transform.resize(image, (128, 128, 3))
            if resized.shape == (128, 128, 3):
                x_train.append((resized - 0.5) * 2)
                iter_y = np.zeros(10)
                np.put(iter_y, [random_folder], [1])
                y_train.append(iter_y)
                amount_loaded += 1
        except ValueError:
            pass
        except IOError:
            pass

    return np.array(x_train), np.array(y_train)


if __name__ == "__main__":
    train_reconstruction(5, 9000, 6)
    # test(5)
