import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io
from sklearn.utils import shuffle
from tensorflow.python.saved_model import tag_constants
from argparse import ArgumentParser
from inception_score import get_inception_score


def encoder(image, attrs, num_attrs):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('reshape'):
            # Input Layer
            input_layer = tf.reshape(image, [-1, 128, 128, 3], 'e_input')

            # Reshape input layer
            a = tf.reshape(attrs, [-1, 1, 1, num_attrs])
            a = tf.tile(a, [1, 128, 128, 1])
            input_layer = tf.concat([input_layer, a], axis=-1)

        # Convolution Layer 1
        with tf.variable_scope('conv1'):
            e_conv_1 = tf.layers.conv2d(input_layer, 64, (5, 5), (2, 2), 'same')
            e_conv_1 = tf.layers.batch_normalization(e_conv_1)
            e_conv_1 = tf.nn.relu(e_conv_1)

        # Convolution Layer 2
        with tf.variable_scope('conv2'):
            e_conv_2 = tf.layers.conv2d(e_conv_1, 128, (5, 5), (2, 2), 'same')
            e_conv_2 = tf.layers.batch_normalization(e_conv_2)
            e_conv_2 = tf.nn.relu(e_conv_2)

        # Convolution Layer 3
        with tf.variable_scope('conv3'):
            e_conv_3 = tf.layers.conv2d(e_conv_2, 256, (5, 5), (2, 2), 'same')
            e_conv_3 = tf.layers.batch_normalization(e_conv_3)
            e_conv_3 = tf.nn.relu(e_conv_3)

        # Convolution Layer 4
        with tf.variable_scope('conv4'):
            e_conv_4 = tf.layers.conv2d(e_conv_3, 512, (5, 5), (2, 2), 'same')
            e_conv_4 = tf.layers.batch_normalization(e_conv_4)
            e_conv_4 = tf.nn.relu(e_conv_4)

        # Mean
        with tf.variable_scope('average'):
            mean = tf.reduce_mean(e_conv_4, axis=[1, 2])

        # Fully connected layers
        with tf.variable_scope('fc'):
            z_avg = tf.layers.dense(mean, 256)
            z_log_var = tf.layers.dense(mean, 256)

    return z_avg, z_log_var


def generator(z, attrs):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        # Concatenate hair style parameter
        with tf.variable_scope('fc1'):
            w = 16
            g_concat = tf.concat([z, attrs], axis=-1)

            # Fully connected layer
            g_fc = tf.layers.dense(g_concat, w * w * 512)
            g_fc = tf.layers.batch_normalization(g_fc)
            g_fc = tf.nn.relu(g_fc)
            g_fc = tf.reshape(g_fc, [-1, w, w, 512])

        # Deconvolution layer 1
        with tf.variable_scope('conv1'):
            g_deconv_1 = tf.layers.conv2d_transpose(g_fc, 512, (5, 5), (2, 2), 'same')
            g_deconv_1 = tf.layers.batch_normalization(g_deconv_1)
            g_deconv_1 = tf.nn.relu(g_deconv_1)

        # Deconvolution layer 1
        with tf.variable_scope('conv2'):
            g_deconv_2 = tf.layers.conv2d_transpose(g_deconv_1, 256, (5, 5), (2, 2), 'same')
            g_deconv_2 = tf.layers.batch_normalization(g_deconv_2)
            g_deconv_2 = tf.nn.relu(g_deconv_2)

        # Deconvolution layer 1
        with tf.variable_scope('conv3'):
            g_deconv_3 = tf.layers.conv2d_transpose(g_deconv_2, 128, (5, 5), (2, 2), 'same')
            g_deconv_3 = tf.layers.batch_normalization(g_deconv_3)
            g_deconv_3 = tf.nn.relu(g_deconv_3)

        # Deconvolution layer 1
        with tf.variable_scope('conv4'):
            g_deconv_4 = tf.layers.conv2d_transpose(g_deconv_3, 3, (5, 5), (1, 1), 'same')
            g_image = tf.tanh(g_deconv_4)

    return g_image


def discriminator(image):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        # Convolution Layer 1
        with tf.variable_scope('conv1'):
            e_conv_1 = tf.layers.conv2d(image, 64, (5, 5), (2, 2), 'same')
            e_conv_1 = tf.layers.batch_normalization(e_conv_1)
            e_conv_1 = tf.nn.relu(e_conv_1)

        # Convolution Layer 2
        with tf.variable_scope('conv2'):
            e_conv_2 = tf.layers.conv2d(e_conv_1, 128, (5, 5), (2, 2), 'same')
            e_conv_2 = tf.layers.batch_normalization(e_conv_2)
            e_conv_2 = tf.nn.relu(e_conv_2)

        # Convolution Layer 3
        with tf.variable_scope('conv3'):
            e_conv_3 = tf.layers.conv2d(e_conv_2, 256, (5, 5), (2, 2), 'same')
            e_conv_3 = tf.layers.batch_normalization(e_conv_3)
            e_conv_3 = tf.nn.relu(e_conv_3)

        # Convolution Layer 4
        with tf.variable_scope('conv4'):
            e_conv_4 = tf.layers.conv2d(e_conv_3, 512, (5, 5), (2, 2), 'same')
            e_conv_4 = tf.layers.batch_normalization(e_conv_4)
            e_conv_4 = tf.nn.relu(e_conv_4)

        # Mean
        with tf.variable_scope('average'):
            mean = tf.reduce_mean(e_conv_4, axis=[1, 2])

        # Image discrimination
        with tf.variable_scope('fc'):
            f = tf.contrib.layers.flatten(mean)
            y = tf.layers.dense(f, 1)

    return y, f


def recognizer(image, y_size):
    with tf.variable_scope('recognizer', reuse=tf.AUTO_REUSE):
        # Convolution Layer 1
        with tf.variable_scope('conv1'):
            e_conv_1 = tf.layers.conv2d(image, 64, (5, 5), (2, 2), 'same')
            e_conv_1 = tf.layers.batch_normalization(e_conv_1)
            e_conv_1 = tf.nn.relu(e_conv_1)

        # Convolution Layer 2
        with tf.variable_scope('conv2'):
            e_conv_2 = tf.layers.conv2d(e_conv_1, 128, (5, 5), (2, 2), 'same')
            e_conv_2 = tf.layers.batch_normalization(e_conv_2)
            e_conv_2 = tf.nn.relu(e_conv_2)

        # Convolution Layer 3
        with tf.variable_scope('conv3'):
            e_conv_3 = tf.layers.conv2d(e_conv_2, 256, (5, 5), (2, 2), 'same')
            e_conv_3 = tf.layers.batch_normalization(e_conv_3)
            e_conv_3 = tf.nn.relu(e_conv_3)

        # Convolution Layer 4
        with tf.variable_scope('conv4'):
            e_conv_4 = tf.layers.conv2d(e_conv_3, 512, (5, 5), (2, 2), 'same')
            e_conv_4 = tf.layers.batch_normalization(e_conv_4)
            e_conv_4 = tf.nn.relu(e_conv_4)

        # Mean
        with tf.variable_scope('average'):
            mean = tf.reduce_mean(e_conv_4, axis=[1, 2])

        with tf.variable_scope('fc'):
            f = tf.contrib.layers.flatten(mean)
            y = tf.layers.dense(f, y_size)

    return y, f


def train(batch_size, iteration_amount, epoch_amount, weights="weights"):
    # Prediction inputs
    image = tf.placeholder('float32', [None, 128, 128, 3], name="image_input")
    y_input = tf.placeholder('float32', [None, 3], name="y_input")
    decoder_y_input = tf.placeholder('float32', [None, 3], name="decoder_y_input")
    z_rand_input = tf.placeholder('float32', [None, 256], name="z_rand_input")

    # Generate sample
    average, deviation = encoder(image, y_input, 3)
    sample = np.random.uniform(-1, 1, [batch_size, 256])
    z = average + sample * deviation

    # Prediction outputs
    # Encoded
    decoded_image = generator(z, decoder_y_input)
    discriminator_decoded_image, discriminator_decoded_f = discriminator(decoded_image)
    recognizer_decoded_image, recognizer_decoded_f = recognizer(decoded_image, 4)

    # Real
    discriminator_real_image, discriminator_real_f = discriminator(image)
    recognizer_real_image, recognizer_real_f = recognizer(image, 4)

    # Random
    decoded_random = generator(z_rand_input, decoder_y_input)
    discriminator_random_image, discriminator_random_f = discriminator(decoded_random)
    recognizer_random_image, recognizer_random_f = recognizer(decoded_random, 4)

    # Recognizer help
    recognizer_real_aug = tf.concat((y_input, tf.zeros((tf.shape(y_input)[0], 1))), axis=1)
    recognizer_other = tf.concat((tf.zeros_like(y_input),
                                  tf.ones((tf.shape(y_input)[0], 1))), axis=1)

    # Loss
    enc_loss = get_encoder_loss(decoded_image, image)
    kl_loss = get_kl_loss(average, deviation)
    g_loss = get_generator_loss(discriminator_decoded_image, discriminator_random_image,
                                recognizer_decoded_image, recognizer_random_image,
                                recognizer_real_aug)
    d_loss = get_discriminator_loss(discriminator_real_image, discriminator_decoded_image, discriminator_random_image)
    q_loss = get_recognizer_loss(recognizer_real_image, recognizer_decoded_image, recognizer_random_image,
                                 recognizer_real_aug, recognizer_other)

    # Variable list
    enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='recognizer')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # Solvers
    e_solver = tf.train.AdamOptimizer(2e-4).minimize(enc_loss + kl_loss, var_list=enc_vars)
    g_solver = tf.train.AdamOptimizer(2e-4).minimize(g_loss + enc_loss, var_list=g_vars)
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
                print('Iteration:', iteration)
                x_training, y_training = get_training_set(batch_size)
                x_training, y_training = shuffle(x_training, y_training)
                sample_z = np.random.uniform(-1, 1, [batch_size, 256])
                _, _, _, _ = session.run([e_solver, g_solver, d_solver, q_solver],
                                         feed_dict={image: x_training, y_input: y_training, z_rand_input: sample_z,
                                                    decoder_y_input: y_training})
                # output_loss(d_loss, enc_loss, g_loss, image, kl_loss, q_loss, sample_z, session, x_training, y_input,
                #             y_training, z_rand_input)
            single_test(session, decoded_image, image, y_input, decoder_y_input, batch_size, epoch)
            single_random_test(session, decoded_random, z_rand_input, y_input, decoder_y_input, batch_size, epoch)
            inputs = {
                "image_placeholder": image,
                "y_input_placeholder": y_input,
            }
            outputs = {"decoder": decoded_image}
            tf.saved_model.simple_save(session, weights + '/epoch-' + str(epoch), inputs, outputs)


def continue_training(batch_size, iteration_amount, epoch_amount, weights="weights_bak/epoch-27"):
    restored_graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session() as session:
            tf.saved_model.loader.load(
                session,
                [tag_constants.SERVING],
                weights,
            )

            e_solver = restored_graph.get_operation_by_name("Adam")
            g_solver = restored_graph.get_operation_by_name("Adam_1")
            d_solver = restored_graph.get_operation_by_name("Adam_2")
            q_solver = restored_graph.get_operation_by_name("Adam_3")

            image = restored_graph.get_tensor_by_name('image_input:0')
            y_input = restored_graph.get_tensor_by_name('y_input:0')
            decoder_y_input = restored_graph.get_tensor_by_name('decoder_y_input:0')
            z_rand_input = restored_graph.get_tensor_by_name('z_rand_input:0')
            decoded_image = restored_graph.get_tensor_by_name('generator/conv4/Tanh:0')
            decoded_random = restored_graph.get_tensor_by_name('generator_1/conv4/Tanh:0')

            # Epochs
            for epoch in range(epoch_amount):
                for iteration in range(1, iteration_amount):
                    x_training, y_training = get_training_set(batch_size)
                    x_training, y_training = shuffle(x_training, y_training)
                    sample_z = np.random.uniform(-1, 1, [batch_size, 256])
                    _, _, _, _ = session.run([e_solver, g_solver, d_solver, q_solver],
                                             feed_dict={image: x_training, y_input: y_training, z_rand_input: sample_z,
                                                        decoder_y_input: y_training})
                single_test(session, decoded_image, image, y_input, decoder_y_input, batch_size, epoch)
                single_random_test(session, decoded_random, z_rand_input, y_input, decoder_y_input, batch_size, epoch)
                inputs = {
                    "image_placeholder": image,
                    "y_input_placeholder": y_input,
                }
                outputs = {"decoder": decoded_image}
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
    encode_decode_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(image, decoder_encoded), axis=[1, 2, 3]))
    return encode_decode_loss


def get_kl_loss(average, deviation):
    # Latent space regularization
    kl_div_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + deviation - tf.square(average) - tf.exp(deviation), axis=-1))
    return kl_div_loss


def get_generator_loss(discriminator_decoded_image, discriminator_random_image,
                       recognizer_decoded_image, recognizer_random_image,
                       recognizer_real_aug):
    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_decoded_image), discriminator_decoded_image)
    g_loss += tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_random_image), discriminator_random_image)
    g_loss += tf.losses.softmax_cross_entropy(recognizer_real_aug, recognizer_decoded_image)
    g_loss += tf.losses.softmax_cross_entropy(recognizer_real_aug, recognizer_random_image)
    return g_loss


def get_discriminator_loss(discriminator_real_image, discriminator_decoded_image, discriminator_random_image):
    d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_real_image), discriminator_real_image)
    d_loss += tf.losses.sigmoid_cross_entropy(tf.zeros_like(discriminator_decoded_image), discriminator_decoded_image)
    d_loss += tf.losses.sigmoid_cross_entropy(tf.zeros_like(discriminator_random_image), discriminator_random_image)
    return d_loss


def get_recognizer_loss(recognizer_real_image, recognizer_decoded_image, recognizer_random_image,
                        recognizer_real_aug, recognizer_other):
    q_loss = tf.losses.softmax_cross_entropy(recognizer_real_aug, recognizer_real_image)
    q_loss += tf.losses.softmax_cross_entropy(recognizer_other, recognizer_decoded_image)
    q_loss += tf.losses.softmax_cross_entropy(recognizer_other, recognizer_random_image)
    return q_loss


def single_test(session, decoder, image_input, y_input, decoder_y_input, batch_size, epoch):
    x_training, y_training = get_training_set(batch_size)
    generated_image = (session.run(decoder, feed_dict={image_input: x_training, y_input: y_training,
                                                       decoder_y_input: y_training}))
    generated_image = (generated_image[0] + 1) / 2 * 255
    generated_image = np.uint8(generated_image)
    generated_image = np.clip(generated_image, 0, 255)
    io.imsave('results/epoch-' + str(epoch) + '.png', generated_image)


def single_random_test(session, decoded_random, z_rand_input, y_input, decoder_y_input, batch_size, epoch):
    _, y_training = get_training_set(batch_size)
    sample = np.random.uniform(-1, 1, [batch_size, 256])
    generated_image = (session.run(decoded_random, feed_dict={z_rand_input: sample, y_input: y_training,
                                                              decoder_y_input: y_training}))
    generated_image = (generated_image[0] + 1) / 2 * 255
    generated_image = np.uint8(generated_image)
    generated_image = np.clip(generated_image, 0, 255)
    io.imsave('results/rand-epoch-' + str(epoch) + '.png', generated_image)


def test(batch_size, weights):
    restored_graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session() as session:
            tf.saved_model.loader.load(
                session,
                [tag_constants.SERVING],
                weights,
            )

            image_input = restored_graph.get_tensor_by_name('image_input:0')
            y_input = restored_graph.get_tensor_by_name('y_input:0')
            decoder_y_input = restored_graph.get_tensor_by_name('decoder_y_input:0')
            decoder = restored_graph.get_tensor_by_name('generator/conv4/Tanh:0')

            x_test, y_test = get_training_set(batch_size)

            generated_image = (session.run(decoder, feed_dict={image_input: x_test, y_input: y_test, decoder_y_input: y_test}))
            generated_image = (generated_image[0] + 1) / 2 * 255
            generated_image = np.uint8(generated_image)
            generated_image = np.clip(generated_image, 0, 255)
            x_test = (x_test[0] + 1) / 2 * 255
            x_test = np.uint8(x_test)
            x_test = np.clip(x_test, 0, 255)

            f = plt.figure()
            plt.title('Reconstruction Test', loc='center')
            plt.xticks([])
            plt.yticks([])
            f.add_subplot(1,2, 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_test)
            f.add_subplot(1,2, 2)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(generated_image)
            plt.show()


def gen_test(batch_size, weights):
    restored_graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session() as session:
            tf.saved_model.loader.load(
                session,
                [tag_constants.SERVING],
                weights,
            )

            image_input = restored_graph.get_tensor_by_name('z_rand_input:0')
            y_input = restored_graph.get_tensor_by_name('y_input:0')
            decoder_y_input = restored_graph.get_tensor_by_name('decoder_y_input:0')
            decoder = restored_graph.get_tensor_by_name('generator_1/conv4/Tanh:0')

            x_test, y_test = get_training_set(batch_size)
            sample = np.random.uniform(-1, 1, [batch_size, 256])

            generated_image = (session.run(decoder, feed_dict={image_input: sample, y_input: y_test, decoder_y_input: y_test}))
            generated_image = (generated_image[0] + 1) / 2 * 255
            generated_image = np.uint8(generated_image)
            generated_image = np.clip(generated_image, 0, 255)

            plt.title('Generation Test')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(generated_image)
            plt.show()


def mod_test(batch_size, weights):
    restored_graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session() as session:
            tf.saved_model.loader.load(
                session,
                [tag_constants.SERVING],
                weights,
            )

            image_input = restored_graph.get_tensor_by_name('image_input:0')
            y_input = restored_graph.get_tensor_by_name('y_input:0')
            decoder_y_input = restored_graph.get_tensor_by_name('decoder_y_input:0')
            decoder = restored_graph.get_tensor_by_name('generator/conv4/Tanh:0')

            new_y = []
            for i in range(10):
                temp_y = np.zeros(3)
                np.put(temp_y, [1], [1])
                new_y.append(temp_y)

            x_test, y_test = get_training_set(batch_size)
            generated_image = (session.run(decoder, feed_dict={image_input: x_test, y_input: y_test, decoder_y_input: new_y}))

            f = plt.figure()
            plt.title('Modification Test', loc='center')
            plt.xticks([])
            plt.yticks([])
            x_test = (x_test[0] + 1) / 2 * 255
            x_test = np.uint8(x_test)
            x_test = np.clip(x_test, 0, 255)
            f.add_subplot(1,2, 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_test)

            generated_image = (generated_image[0] + 1) / 2 * 255
            generated_image = np.uint8(generated_image)
            generated_image = np.clip(generated_image, 0, 255)
            f.add_subplot(1,2, 2)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(generated_image)
            plt.show()


def inception_score(batch_size=10, weights="weights/epoch-27"):
    generated_images = []
    restored_graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session() as session:
            tf.saved_model.loader.load(
                session,
                [tag_constants.SERVING],
                weights,
            )

            image_input = restored_graph.get_tensor_by_name('z_rand_input:0')
            y_input = restored_graph.get_tensor_by_name('y_input:0')
            decoder_y_input = restored_graph.get_tensor_by_name('decoder_y_input:0')
            decoder = restored_graph.get_tensor_by_name('generator_1/conv4/Tanh:0')

            for i in range(64):
                y_test = []
                for i in range(batch_size):
                    random_folder = random.randint(0, 2)
                    iter_y = np.zeros(3)
                    np.put(iter_y, [random_folder], [1])
                    y_test.append(iter_y)
                sample = np.random.uniform(-1, 1, [batch_size, 256])

                generated_image = (session.run(decoder, feed_dict={image_input: sample, y_input: y_test,
                                                                   decoder_y_input: y_test}))
                if generated_images == []:
                    generated_images = generated_image
                else:
                    generated_images = np.append(generated_images, generated_image, axis=0)
                    generated_images = (generated_images + 1) / 2 * 255
                    generated_images = np.uint8(generated_images)
                    generated_images = np.clip(generated_images, 0, 255)
    generated_images = np.reshape(generated_images, (generated_images.shape[0], 3, 128, 128))
    print(get_inception_score(generated_images))


def __color_to_enum(color):
    if color == "black":
        color = 0
    elif color == "blonde":
        color = 1
    else:
        color = 2
    return color


def change_hair_color(image, color, new_color, batch_size=10, weights="weights/epoch-27"):
    color = __color_to_enum(color)
    new_color = __color_to_enum(new_color)

    restored_graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session() as session:
            tf.saved_model.loader.load(
                session,
                [tag_constants.SERVING],
                weights,
            )

            image_input = restored_graph.get_tensor_by_name('image_input:0')
            y_input = restored_graph.get_tensor_by_name('y_input:0')
            decoder_y_input = restored_graph.get_tensor_by_name('decoder_y_input:0')
            decoder = restored_graph.get_tensor_by_name('generator/conv4/Tanh:0')

            new_y = []
            x_test = []
            y_test = []

            for i in range(10):
                temp_y = np.zeros(3)
                np.put(temp_y, [new_color], [1])
                new_y.append(temp_y)

                read_image = io.imread(image)
                resized = transform.resize(read_image, (128, 128, 3))
                x_test.append((resized - 0.5) * 2)

                iter_y = np.zeros(3)
                np.put(iter_y, [color], [1])
                y_test.append(iter_y)

            generated_image = (session.run(decoder, feed_dict={image_input: x_test, y_input: y_test, decoder_y_input: new_y}))

            f = plt.figure()
            x_test = (x_test[0] + 1) / 2 * 255
            x_test = np.uint8(x_test)
            x_test = np.clip(x_test, 0, 255)
            f.add_subplot(1, 2, 1)
            plt.imshow(x_test)

            generated_image = (generated_image[0] + 1) / 2 * 255
            generated_image = np.uint8(generated_image)
            generated_image = np.clip(generated_image, 0, 255)
            f.add_subplot(1, 2, 2)
            plt.imshow(generated_image)
            plt.show()


def multiple_mod_test(batch_size, new_color, weights):
    new_color = __color_to_enum(new_color)
    restored_graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session() as session:
            tf.saved_model.loader.load(
                session,
                [tag_constants.SERVING],
                weights,
            )

            image_input = restored_graph.get_tensor_by_name('image_input:0')
            y_input = restored_graph.get_tensor_by_name('y_input:0')
            decoder_y_input = restored_graph.get_tensor_by_name('decoder_y_input:0')
            decoder = restored_graph.get_tensor_by_name('generator/conv4/Tanh:0')

            new_y = []
            for i in range(10):
                temp_y = np.zeros(3)
                np.put(temp_y, [new_color], [1])
                new_y.append(temp_y)

            x_tests, y_test = get_training_set(batch_size)
            generated_images = (session.run(decoder, feed_dict={image_input: x_tests, y_input: y_test,
                                                                decoder_y_input: new_y}))

            for i in range(batch_size):
                f = plt.figure()
                plt.title('Modification Test', loc='center')
                plt.xticks([])
                plt.yticks([])
                x_test = (x_tests[i] + 1) / 2 * 255
                x_test = np.uint8(x_test)
                x_test = np.clip(x_test, 0, 255)
                f.add_subplot(1, 2, 1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(x_test)

                generated_image = (generated_images[i] + 1) / 2 * 255
                generated_image = np.uint8(generated_image)
                generated_image = np.clip(generated_image, 0, 255)
                f.add_subplot(1, 2, 2)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(generated_image)
                plt.show()


def get_training_set(load_amount):
    x_train = []
    y_train = []
    amount_loaded = 1
    while amount_loaded <= load_amount:
        random_folder = random.randint(0, 2)
        folder = "/home/ilias/Repositories/hairy_gan/celeba-dataset/" + str(random_folder)
        files = os.listdir(folder)
        try:
            image = io.imread(os.path.join(folder, random.choice(files)))
            resized = transform.resize(image, (128, 128, 3))
            if resized.shape == (128, 128, 3):
                x_train.append((resized - 0.5) * 2)
                iter_y = np.zeros(3)
                np.put(iter_y, [random_folder], [1])
                y_train.append(iter_y)
                amount_loaded += 1
        except ValueError:
            pass
        except IOError:
            pass

    return np.array(x_train), np.array(y_train)


parser = ArgumentParser(description='Change hair color of an image via a CVAEGAN.')
parser.add_argument("-i", "--image", dest="image",
                    help="Location of the image in JPEG format", metavar="IMAGE",
                    type=str, default="")
parser.add_argument("-t", "--train", dest="train",
                    help="Trains the model",
                    default=False, action='store_true')
parser.add_argument("-mult", "--mult_test", dest="mult_test",
                    help="Test multiple modifications",
                    default=False, action='store_true')
parser.add_argument("-test", "--testing", dest="test",
                    help="Run construction, generation and modification tests",
                    default=False, action='store_true')
parser.add_argument("-score", "--inception_score", dest="test_score",
                    help="Run inception_score",
                    default=False, action='store_true')
parser.add_argument("-c", "--continue", dest="cont",
                    help="Continue training a model",
                    default=False, action='store_true')
parser.add_argument("-b", "--batch", dest="batch",
                    help="The amount of batches to use", metavar="BATCH",
                    type=int, default=10)
parser.add_argument("-e", "--epoch", dest="epoch",
                    help="The amount of epoch to train", metavar="EPOCH",
                    type=int, default=28)
parser.add_argument("-iter", "--iterations", dest="iterations",
                    help="The amount of iterations to train", metavar="ITER",
                    type=int, default=3000)
parser.add_argument("-w", "--weights", dest="weights",
                    help="The location of the weights", metavar="WEIGHT",
                    type=str, default="")
parser.add_argument("-col", "--color", dest="color",
                    help="Trains the model", metavar="COLOR",
                    type=str, default="")
parser.add_argument("-or", "--original_color", dest="original_color",
                    help="Trains the model", metavar="ORIGINAL_COLOR",
                    type=str, default="")
args = parser.parse_args()


if __name__ == "__main__":
    if args.train:
        if args.weights == "":
            print("Please provide a location for the weights")
        else:
            print("Starting training...")
            train(args.batch, args.iterations, args.epoch, args.weights)
    elif args.test_score:
        if args.weights == "":
            print("Please provide a location for the weights")
        else:
            print("Testing inception score...")
            inception_score(args.batch, args.weights)
    elif args.cont:
        if args.weights == "":
            print("Please provide a location for the weights")
        else:
            print("Continuing training...")
            continue_training(args.batch, args.iterations, args.epoch, args.weights)
    elif args.test:
        if args.weights == "":
            print("Please provide a location for the weights")
        else:
            print("Starting tests...")
            # test(args.batch, args.weights)
            # gen_test(args.batch, args.weights)
            mod_test(args.batch, args.weights)
    elif args.mult_test:
        if args.weights == "":
            print("Please provide a location for the weights")
        elif args.color == "":
            print("Please provide the new hair color in the image")
        else:
            print("Starting multiple modification tests...")
            # test(args.batch, args.weights)
            # gen_test(args.batch, args.weights)
            multiple_mod_test(args.batch, args.color, args.weights)
    else:
        if args.image == "":
            print("Please provide the image location")
        elif args.color == "":
            print("Please provide the new hair color in the image")
        elif args.original_color == "":
            print("Please provide the hair color to change tp")
        else:
            change_hair_color(args.image, args.original_color, args.color)
    # train(10, 3000, 28)
    # continue_training(10, 3000, 400)
    # test(10)
    # gen_test(10)
    # mod_test(10, "weights/epoch-27")
    # change_hair_color("zimcke.jpg", "blonde", "black")
