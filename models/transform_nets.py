import tensorflow as tf
import numpy as np
import models.tf_util as tf_util


def input_transform_net(point_cloud, is_training, bn_decay=None, K=3, dtype=tf.float32, bn=False):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    num_feature = point_cloud.get_shape()[2].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,K],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay,
                         dtype=dtype)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay,
                         dtype=dtype)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay,
                         dtype=dtype)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=bn, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay,
                                  dtype=dtype)
    net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay,
                                  dtype=dtype)

    with tf.compat.v1.variable_scope('transform_XYZ'):
        assert K == num_feature,f'K={K}  num_feature={num_feature}'
        weights = tf.compat.v1.get_variable('weights', [256, K * K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=dtype)
        biases = tf.compat.v1.get_variable('biases', [K * K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=dtype)
        biases = biases + tf.constant(np.eye(K).flatten(), dtype=dtype)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64, dtype=tf.float32, bn=False):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value
    num_feature = inputs.get_shape()[2].value

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay,
                         dtype=dtype)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay,
                         dtype=dtype)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay,
                         dtype=dtype)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=bn, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay,
                                  dtype=dtype)
    net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay,
                                  dtype=dtype)

    with tf.compat.v1.variable_scope('transform_feat') as sc:
        weights = tf.compat.v1.get_variable('weights', [256, K * K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=dtype)
        biases = tf.compat.v1.get_variable('biases', [K * K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=dtype)
        biases = biases + tf.constant(np.eye(K).flatten(), dtype=dtype)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
