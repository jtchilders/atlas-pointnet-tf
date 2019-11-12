import tensorflow as tf
import numpy as np
import models.tf_util as tf_util
from models.transform_nets import input_transform_net, feature_transform_net
import logging
logger = logging.getLogger(__name__)


def placeholder_inputs(batch_size, num_point, feature_size):
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, feature_size))
    labels_pl = tf.placeholder(tf.int32,
                               shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, classes, bn_decay=None):
    """ Classification PointNet, input is BxNxF, output BxNxC
        B = number of images per batch
        N = number of points
        F = number of features
        C = number of classes """
    # batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    features = point_cloud.get_shape()[2].value
    end_points = {}

    with tf.variable_scope('transform_net1'):
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=features)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,features],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2'):
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])
    logger.info('point_feat = %s',point_feat)

    net = tf_util.conv2d(point_feat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')
    logger.info('global_feat = %s',global_feat)

    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand],3)
    logger.info('concat_feat = %s',concat_feat)

    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tf_util.conv2d(net, classes, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    net = tf.squeeze(net, [2])  # BxNxC
    logger.info('net = %s',net)

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform']  # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat_loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


def get_accuracy(pred,label,classes=2):

   label_onehot = tf.one_hot(label,classes,axis=-1)
   return IoU_coeff(pred,label_onehot)


def IoU_coeff(pred,label,smooth=1,point_axis=1):
   intersection = tf.math.reduce_sum(tf.abs(label * pred),axis=point_axis)  # BxC
   # logger.info(' intersection = %s ',intersection)
   union = tf.math.reduce_sum(label,axis=point_axis) + tf.math.reduce_sum(pred,axis=point_axis) - intersection
   # logger.info(' union = %s ',union)
   iou = tf.math.reduce_sum((intersection + smooth) / (union + smooth), axis=0)
   for i in range(iou.get_shape()[0].value):
      tf.summary.scalar('class_acc_%i' % i,iou[i])
   return tf.math.reduce_sum(iou)


def get_learning_rate(batch,config):
    learning_rate = tf.train.exponential_decay(
                        config['optimizer']['lr'],  # Base learning rate.
                        batch * config['data']['batch_size'],  # Current index into the dataset.
                        config['optimizer']['step_size'],          # Decay step.
                        config['optimizer']['decay'],          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
