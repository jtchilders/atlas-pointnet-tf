import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
import numpy as np
import models.tf_util as tf_util
from models.transform_nets import input_transform_net, feature_transform_net
import logging
logger = logging.getLogger(__name__)


num_classes = None


def placeholder_inputs(batch_size, num_point, feature_size):
    pointclouds_pl =  tf.compat.v1.placeholder(tf.float32,
                              shape=(batch_size, num_point, feature_size))
    labels_pl =  tf.compat.v1.placeholder(tf.int32,
                              shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, classes, bn_decay=None):
    """ Classification PointNet, input is BxNxF, output BxNxC
        B = number of images per batch
        N = number of points
        F = number of features
        C = number of classes """
    # batch_size = point_cloud.get_shape()[0].value
    global num_classes
    num_classes = classes
    num_point = point_cloud.get_shape()[1].value
    features = point_cloud.get_shape()[2].value
    end_points = {}

    with tf.compat.v1.variable_scope('transform_net1'):
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

    with tf.compat.v1.variable_scope('transform_net2'):
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


def get_loss(pred, label, end_points, reg_weight=0.001,gamma=2):

   return focal_loss_softmax(pred,label,end_points,gamma)


def get_loss_orig(pred, label, end_points, reg_weight=0.001):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.compat.v1.summary.scalar('loss/classify', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform']  # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.compat.v1.summary.scalar('loss/mat', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


def get_loss_sampling(pred,label, end_points, reg_weight=0.001):
   """ pred: BxNxC,
        label: BxN, """
   something_mask = tf.cast(tf.equal(label,tf.fill(label.get_shape(),1)),tf.float32)
   nothing_mask = tf.cast(tf.equal(label,tf.fill(label.get_shape(),0)),tf.float32)

   num_points_per_class = tf.cast(tf.reduce_sum(something_mask,axis=[0,1]),tf.int32)
   logger.info('nothing_mask = %s',nothing_mask)
   logger.info('num_points_per_class = %s',num_points_per_class)
   nothing_mask_reduced = tf.compat.v1.numpy_function(np_sample_mask,[nothing_mask,num_points_per_class],tf.float32)

   loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)

   something_loss = loss * something_mask  # BxN
   nothing_loss = loss * nothing_mask_reduced  # BxN

   loss_value = something_loss + nothing_loss  # BxN
   return tf.reduce_mean(loss_value)


def focal_loss_softmax(pred,label,end_points,gamma=2):
    """
    https://github.com/fudannlp16/focal-loss/blob/master/focal_loss.py
    Computer focal loss for multi classification
    Args:
      label: A int32 tensor of shape [batch_size,points].
      pred: A float32 tensor of shape [batch_size,points,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.softmax(pred,dim=-1)  # [batch_size,points,num_classes]
    label = tf.one_hot(label,depth=y_pred.shape[2])  # [batch_size,points,num_classes]
    L = -label * ((1 - y_pred) ** gamma) * tf.log(y_pred)  # [batch_size,points,num_classes]
    L = tf.reduce_sum(L,axis=[1,2])  # [batch_size]
    return tf.reduce_mean(L)


def np_sample_mask(nothing_mask,nsample):
   nothing_indices = np.nonzero(nothing_mask)
   range_indices = np.arange(len(nothing_indices[0]))
   range_indices_reduced = np.random.choice(range_indices,nsample)
   nothing_indices_reduced = []
   for i in range(len(nothing_indices)):
      nothing_indices_reduced.append(nothing_indices[i][range_indices_reduced])
   nothing_indices_reduced = tuple(nothing_indices_reduced)

   new_mask = np.zeros(nothing_mask.shape,dtype=np.float32)
   new_mask[nothing_indices_reduced] = nothing_mask[nothing_indices_reduced]
   return new_mask


def get_accuracy(pred,label):

   # pred shape [batch,points,classes]
   # label shape [batch,points]

   pred = tf.nn.softmax(pred)
   label_onehot = tf.one_hot(label,num_classes,axis=-1)
   return IoU_coeff(pred,label_onehot)


def IoU_coeff(pred,label,smooth=1,point_axis=1):
   intersection = tf.math.reduce_sum(tf.abs(label * pred),axis=point_axis)  # BxC
   # tf.print(intersection,output_stream=sys.stderr)
   union = tf.math.reduce_sum(label,axis=point_axis) + tf.math.reduce_sum(pred,axis=point_axis) - intersection
   # tf.print(union,output_stream=sys.stderr)
   iou = tf.math.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
   for i in range(iou.get_shape()[0].value):
      tf.compat.v1.summary.scalar('accuracy/class_%i' % i,iou[i])
   return tf.math.reduce_mean(iou)


def get_learning_rate(batch,config):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        config['optimizer']['lr'],             # Base learning rate.
                        batch * config['data']['batch_size'],  # Current index into the dataset.
                        config['optimizer']['step_size'],      # Decay step.
                        config['optimizer']['decay'],          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, config['optimizer']['lr_min']) # CLIP THE LEARNING RATE!
    return learning_rate


def cyclic_learning_rate(global_step,config):
                         
   # https://github.com/mhmoodlan/cyclic-learning-rate/blob/master/clr.py
   """Applies cyclic learning rate (CLR).
     From the paper:
     Smith, Leslie N. "Cyclical learning
     rates for training neural networks." 2017.
     [https://arxiv.org/pdf/1506.01186.pdf]
      This method lets the learning rate cyclically
     vary between reasonable boundary values
     achieving improved classification accuracy and
     often in fewer iterations.
      This code varies the learning rate linearly between the
     minimum (learning_rate) and the maximum (max_lr).
      It returns the cyclic learning rate. It is computed as:
       ```python
       cycle = floor( 1 + global_step /
        ( 2 * step_size ) )
      x = abs( global_step / step_size – 2 * cycle + 1 )
      clr = learning_rate +
        ( max_lr – learning_rate ) * max( 0 , 1 - x )
       ```
      Polices:
        'triangular':
          Default, linearly increasing then linearly decreasing the
          learning rate at each cycle.
         'triangular2':
          The same as the triangular policy except the learning
          rate difference is cut in half at the end of each cycle.
          This means the learning rate difference drops after each cycle.
         'exp_range':
          The learning rate varies between the minimum and maximum
          boundaries and each boundary value declines by an exponential
          factor of: gamma^global_step.
       Example: 'triangular2' mode cyclic learning rate.
        '''python
        ...
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=
          clr.cyclic_learning_rate(global_step=global_step, mode='triangular2'))
        train_op = optimizer.minimize(loss_op, global_step=global_step)
        ...
         with tf.Session() as sess:
            sess.run(init)
            for step in range(1, num_steps+1):
              assign_op = global_step.assign(step)
              sess.run(assign_op)
        ...
         '''
       Args:
        global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
          Global step to use for the cyclic computation.  Must not be negative.
        learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate which is the lower bound
          of the cycle (default = 0.1).
        max_lr:  A scalar. The maximum learning rate boundary.
        step_size: A scalar. The number of iterations in half a cycle.
          The paper suggests step_size = 2-8 x training iterations in epoch.
        gamma: constant in 'exp_range' mode:
          gamma**(global_step)
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
        name: String.  Optional name of the operation.  Defaults to
          'CyclicLearningRate'.
       Returns:
        A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
        learning rate.
      Raises:
        ValueError: if `global_step` is not supplied.
      @compatibility(eager)
      When eager execution is enabled, this function returns
      a function which in turn returns the decayed learning
      rate Tensor. This can be useful for changing the learning
      rate value across different invocations of optimizer functions.
      @end_compatibility
   """
   learning_rate  = config['optimizer']['lr']
   max_lr         = config['optimizer']['max_lr']
   step_size      = config['optimizer']['step_size']
   gamma          = config['optimizer']['decay']
   mode           = config['optimizer']['mode']
   name           = None
   if global_step is None:
      raise ValueError("global_step is required for cyclic_learning_rate.")
   with ops.name_scope(name, "CyclicLearningRate",
                       [learning_rate, global_step]) as name:
      learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
      dtype = learning_rate.dtype
      global_step = math_ops.cast(global_step, dtype)
      step_size = math_ops.cast(step_size, dtype)
      
      def cyclic_lr():
         """Helper to recompute learning rate; most helpful in eager-mode."""
         # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
         double_step = math_ops.multiply(2., step_size)
         global_div_double_step = math_ops.divide(global_step, double_step)
         cycle = math_ops.floor(math_ops.add(1., global_div_double_step))
         # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
         double_cycle = math_ops.multiply(2., cycle)
         global_div_step = math_ops.divide(global_step, step_size)
         tmp = math_ops.subtract(global_div_step, double_cycle)
         x = math_ops.abs(math_ops.add(1., tmp))
         # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
         a1 = math_ops.maximum(0., math_ops.subtract(1., x))
         a2 = math_ops.subtract(max_lr, learning_rate)
         clr = math_ops.multiply(a1, a2)
         if mode == 'triangular2':
            clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
                                  cycle - 1, tf.int32)), tf.float32))
         if mode == 'exp_range':
            clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)
         return math_ops.add(clr, learning_rate, name=name)
      if not context.executing_eagerly():
         cyclic_lr = cyclic_lr()
      return cyclic_lr
