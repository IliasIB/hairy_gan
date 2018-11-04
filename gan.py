import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.nn import conv2d
from skimage import transform, io


def encoder(image):
    with tf.variable_scope('encoder') as scope:
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


def generator(average, deviation, sample_dimension):
    with tf.variable_scope('generator') as scope:
        # Generate sample
        sample = np.random.uniform(-1, 1, [1, sample_dimension])

        # Calculate new sample
        z = average + sample * deviation

        # Fully connected layer 1
        w_fc_1 = tf.get_variable('g_wfc1', [sample_dimension, 8064])
        b_fc_1 = tf.get_variable('g_bfc1', [8064])
        h_fc_1 = tf.nn.relu(tf.matmul(z, w_fc_1) + b_fc_1)

        # Fully connected layer 2
        w_fc_2 = tf.get_variable('g_wfc2', [8064, 4 * 4 * 72])
        b_fc_2 = tf.get_variable('g_bfc2', [4 * 4 * 72])
        h_fc_2 = tf.nn.relu(tf.matmul(h_fc_1, w_fc_2) + b_fc_2)

        # Hidden Layer
        hidden_layer = tf.reshape(h_fc_2, [1, 4, 4, 72])
        hidden_layer = tf.nn.relu(hidden_layer)

        # DeConv Layer 1
        w_conv1 = tf.get_variable('g_wconv1', [3, 3, 288, 72])
        b_conv1 = tf.get_variable('g_bconv1', [288])
        h_conv1 = tf.nn.conv2d_transpose(hidden_layer, w_conv1, output_shape=[1, 8, 8, 288],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        h_conv1 = tf.contrib.layers.batch_norm(inputs=h_conv1, scale=True, scope="g_bn1")
        h_conv1 = tf.nn.relu(h_conv1)

        # DeConv Layer 2
        w_conv2 = tf.get_variable('g_wconv2', [3, 3, 216, 288])
        b_conv2 = tf.get_variable('g_bconv2', [216])
        h_conv2 = tf.nn.conv2d_transpose(h_conv1, w_conv2, output_shape=[1, 16, 16, 216],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv2
        h_conv2 = tf.contrib.layers.batch_norm(inputs=h_conv2, scale=True, scope="g_bn2")
        h_conv2 = tf.nn.relu(h_conv2)

        # DeConv Layer 3
        w_conv3 = tf.get_variable('g_wconv3', [5, 5, 144, 216])
        b_conv3 = tf.get_variable('g_bconv3', [144])
        h_conv3 = tf.nn.conv2d_transpose(h_conv2, w_conv3, output_shape=[1, 32, 32, 144],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv3
        h_conv3 = tf.contrib.layers.batch_norm(inputs=h_conv3, scale=True, scope="g_bn3")
        h_conv3 = tf.nn.relu(h_conv3)

        # DeConv Layer 4
        w_conv4 = tf.get_variable('g_wconv4', [5, 5, 72, 144])
        b_conv4 = tf.get_variable('g_bconv4', [72])
        h_conv4 = tf.nn.conv2d_transpose(h_conv3, w_conv4, output_shape=[1, 64, 64, 72],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv4
        h_conv4 = tf.contrib.layers.batch_norm(inputs=h_conv4, scale=True, scope="g_bn4")
        h_conv4 = tf.nn.relu(h_conv4)

        # DeConv Layer 5
        w_conv5 = tf.get_variable('g_wconv5', [6, 6, 3, 72])
        b_conv5 = tf.get_variable('g_bconv5', [3])
        h_conv5 = tf.nn.conv2d_transpose(h_conv4, w_conv5, output_shape=[1, 128, 128, 3],
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv5
        h_conv5 = tf.nn.tanh(h_conv5)
    return h_conv5


def discrimator(image):
    with tf.variable_scope('discriminator') as scope:
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
    return h_fc


def test():
    with tf.Session() as session:
        img = io.imread('Tenkind/6bald/1.jpg')
        resized = transform.resize(img, (128, 128, 3))
        resized = np.reshape(resized, (1, 128, 128, 3))

        # plt.imshow(resized)
        # plt.show()

        image_input = tf.placeholder('float32', [None, 128, 128, 3], name="image_input")
        average, deviation = encoder(image_input)
        sample_image = generator(average, deviation, 256)

        session.run(tf.global_variables_initializer())
        generated_image = (session.run(sample_image, feed_dict={image_input: resized}))

        # my_i = generated_image.squeeze()
        generated_image = np.reshape(generated_image, (128, 128, 3))
        plt.imshow(generated_image)
        plt.show()


if __name__ == "__main__":
    test()
