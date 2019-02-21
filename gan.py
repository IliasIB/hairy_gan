import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io
from sklearn.utils import shuffle
from tensorflow.python.saved_model import tag_constants


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
            print(g_image)

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


def train_reconstruction(batch_size, iteration_amount, epoch_amount):
    # Prediction inputs
    image = tf.placeholder('float32', [None, 128, 128, 3], name="reconstruction_training_image_input")
    y_input = tf.placeholder('float32', [None, 10], name="reconstruction_training_y_input")
    z_rand_input = tf.placeholder('float32', [None, 256], name="reconstruction_training_z_rand_input")

    # Generate sample
    average, deviation = encoder(image, y_input, 10)
    sample = np.random.uniform(-1, 1, [batch_size, 256])
    z = average + sample * deviation

    # Prediction outputs
    # Encoded
    decoded_image = generator(z, y_input)
    discriminator_decoded_image, discriminator_decoded_f = discriminator(decoded_image)
    recognizer_decoded_image, recognizer_decoded_f = recognizer(decoded_image, 10)

    # Real
    discriminator_real_image, discriminator_real_f = discriminator(image)
    recognizer_real_image, recognizer_real_f = recognizer(image, 10)

    # Random
    decoded_random = generator(z_rand_input, y_input)
    discriminator_random_image, discriminator_random_f = discriminator(decoded_random)
    recognizer_random_image, recognizer_random_f = recognizer(decoded_random, 10)

    # Loss
    kl_loss = get_kl_loss(average, deviation)
    g_loss = get_generator_loss(image, decoded_image, discriminator_real_f, discriminator_decoded_f,
                                recognizer_real_f, recognizer_decoded_f)
    d_loss = get_discriminator_loss(discriminator_real_image, discriminator_decoded_image, discriminator_random_image)
    q_loss = get_recognizer_loss(recognizer_real_image, recognizer_decoded_image)
    gd_loss = get_discriminator_feature_matching_loss(discriminator_real_f, discriminator_decoded_f)
    gq_loss = get_recognizer_feature_matching_loss(discriminator_real_f, discriminator_decoded_f)

    # Variable list
    enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='recognizer')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # Weights
    lambda_1 = 3
    lambda_2 = 1
    lambda_3 = 1e-3
    lambda_4 = 1e-3

    # Solvers
    e_solver = tf.train.AdamOptimizer(2e-4).minimize(lambda_2 * g_loss + lambda_1 * kl_loss, var_list=enc_vars)
    g_solver = tf.train.AdamOptimizer(2e-4).minimize(lambda_2 * g_loss +
                                                     lambda_3 * gd_loss + lambda_4 * gq_loss, var_list=g_vars)
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
                                         feed_dict={image: x_training, y_input: y_training, z_rand_input: sample_z})
                # output_loss(d_loss, enc_loss, g_loss, image, kl_loss, q_loss, sample_z, session, x_training, y_input,
                #             y_training, z_rand_input)

            # single_test(session, decoder_encoded, image, y_input, batch_size)
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
    encode_decode_loss = 0.5 * tf.losses.mean_squared_error(labels=image, predictions=decoder_encoded)
    return encode_decode_loss


def get_kl_loss(average, deviation):
    # Latent space regularization
    kl_div_loss = 1 + deviation - tf.square(average) - tf.exp(deviation)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    return tf.reduce_mean(kl_div_loss)


def get_generator_loss(real_image, decoded_image, discriminator_real_f, discriminator_decoded_f,
                       recognizer_real_f, recognizer_decoded_f):
    g_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(real_image,
                                                                      decoded_image), axis=[1, 2, 3]))
    g_loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(discriminator_real_f,
                                                                       discriminator_decoded_f), axis=[1]))
    g_loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(recognizer_real_f,
                                                                       recognizer_decoded_f), axis=[1]))
    return g_loss


def get_discriminator_loss(discriminator_real_image, discriminator_decoded_image, discriminator_random_image):
    d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_real_image), discriminator_real_image)
    d_loss += tf.losses.sigmoid_cross_entropy(tf.zeros_like(discriminator_decoded_image), discriminator_decoded_image)
    d_loss += tf.losses.sigmoid_cross_entropy(tf.zeros_like(discriminator_random_image), discriminator_random_image)
    return d_loss


def get_recognizer_loss(recognizer_real_image, recognizer_decoded_image):
    q_loss = tf.losses.softmax_cross_entropy(recognizer_real_image, recognizer_decoded_image)
    return q_loss


def get_discriminator_feature_matching_loss(discriminator_real_f, discriminator_decoded_f):
    expected_real_features = tf.reduce_mean(discriminator_real_f, axis=0)
    expected_fake_features = tf.reduce_mean(discriminator_decoded_f, axis=0)
    return 0.5 * tf.losses.mean_squared_error(expected_real_features, expected_fake_features)


def get_recognizer_feature_matching_loss(recognizer_real_f, recognizer_decoded_f):
    expected_real_features = tf.reduce_mean(recognizer_real_f, axis=0)
    expected_fake_features = tf.reduce_mean(recognizer_decoded_f, axis=0)
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

            decoder = restored_graph.get_tensor_by_name('generator/conv4/Tanh:0')

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
    train_reconstruction(10, 3000, 6)
    # test(10)
