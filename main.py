#!/usr/bin/env python
import argparse, logging, json, os, time

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_handler
from models import pointnet_seg

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = 'config.json'
DEFAULT_INTEROP = 1
DEFAULT_INTRAOP = os.cpu_count()
DEFAULT_LOGDIR = '/tmp/tf-' + str(os.getpid())

tf.debugging.set_log_device_placement(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.client import device_lib


def main():
   ''' simple starter program for tensorflow models. '''

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c', '--config', dest='config_filename',
                       help='configuration filename in json format [default: %s]' % DEFAULT_CONFIG,
                       default=DEFAULT_CONFIG)
   parser.add_argument('--interop',
                       help='set Tensorflow "inter_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTEROP,
                       default=DEFAULT_INTEROP)
   parser.add_argument('--intraop',
                       help='set Tensorflow "intra_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTRAOP,
                       default=DEFAULT_INTRAOP)
   parser.add_argument('-l', '--logdir', default=DEFAULT_LOGDIR,
                       help='define location to save log information [default: %s]' % DEFAULT_LOGDIR)

   parser.add_argument('-r', '--restore', help='define location from which to load a saved model')

   parser.add_argument('--horovod', default=False, action='store_true', help="use horovod for MPI parallel training")
   parser.add_argument('--float16', default=False, action='store_true', help="use float16 precision training")

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename', dest='logfilename', default=None,
                       help='if set, logging information will go to file')
   args = parser.parse_args()

   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   hvd = None
   rank = 0
   nranks = 1
   local_rank = 0
   local_nranks = 1
   logging_level = logging.INFO
   if args.horovod:
      import horovod
      import horovod.tensorflow as hvd
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      local_rank = hvd.local_rank()
      local_nranks = hvd.local_size()
      os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
      logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:' + (
               '%05d' % rank) + ':%(name)s:%(message)s'
      if rank > 0:
         logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   logging.warning('rank: %5d   size: %5d  local rank: %5d  local size: %5d', rank, nranks, local_rank, local_nranks)
   if 'CUDA_VISIBLE_DEVICES' in os.environ:
      logging.warning('CUDA_VISIBLE_DEVICES=%s %s', os.environ['CUDA_VISIBLE_DEVICES'], device_lib.list_local_devices())
   else:
      logging.info('CUDA_VISIBLE_DEVICES not defined in os.environ')
   logging.info('using tensorflow version:   %s', tf.__version__)
   logging.info('using tensorflow from:      %s', tf.__file__)
   if hvd:
      logging.info('using horovod version:      %s', horovod.__version__)
      logging.info('using horovod from:         %s', horovod.__file__)
   logging.info('logdir:                     %s', args.logdir)
   logging.info('interop:                    %s', args.interop)
   logging.info('intraop:                    %s', args.intraop)
   logging.info('restore:                    %s', args.restore)
   logging.info('float16:                    %s', args.float16)

   device_str = '/CPU:0'
   if tf.test.is_gpu_available():
      gpus = tf.config.experimental.list_logical_devices('GPU')
      logger.warning('gpus = %s', gpus)

      device_str = gpus[0].name

   logger.warning('device:                     %s', device_str)

   config = json.load(open(args.config_filename))
   config['device'] = device_str
   config['float16'] = args.float16

   nclasses = len(config['data']['classes'])
   dtype_input = tf.float16 if config['float16'] else tf.float32
   dtype_target = tf.int16 if config['float16'] else tf.int32
   bn = True if config['data']['batch_size'] > 1 else False

   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   logger.info('%s = \n %s', args.config_filename, json.dumps(config, indent=4, sort_keys=True))
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   if hvd:
      config['hvd'] = hvd
   config['rank'] = rank
   config['nranks'] = nranks

   with tf.Graph().as_default():

      logger.info('getting datasets')
      trainds, validds = data_handler.get_datasets(config)

      iterator = tf.compat.v1.data.Iterator.from_structure((dtype_input, dtype_target), (
      (config['data']['batch_size'], config['data']['num_points'], config['data']['num_features']),
      (config['data']['batch_size'], config['data']['num_points'])))
      input, target = iterator.get_next()
      training_init_op = iterator.make_initializer(trainds)
      validation_init_op = iterator.make_initializer(validds)

      with tf.device(device_str):

         # input, target = pointnet_seg.placeholder_inputs(config['data']['batch_size'],
         #                                                 config['data']['num_points'],
         #                                                 config['data']['num_features'])

         # handle = tf.compat.v1.placeholder(tf.string,shape=[])
         # iterator = tf.compat.v1.data.Iterator.from_string_handle(handle,(tf.float32,tf.int32),((config['data']['batch_size'],config['data']['num_points'],config['data']['num_features']),(config['data']['batch_size'],config['data']['num_points'])))
         # input,target = iterator.get_next()

         # iter_train = trainds.make_one_shot_iterator()
         # iter_valid = validds.make_one_shot_iterator()

         is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
         batch = tf.Variable(0)


         pred, endpoints = pointnet_seg.get_model(input, is_training_pl, nclasses, dtype=dtype_input, bn=bn)
         loss = pointnet_seg.get_loss(pred, target, endpoints, dtype=dtype_input)
         tf.compat.v1.summary.scalar('loss/combined', loss)

         accuracy = pointnet_seg.get_accuracy(pred, target, dtype=dtype_input)
         tf.compat.v1.summary.scalar('accuracy/combined', accuracy)

         learning_rate = pointnet_seg.cyclic_learning_rate(batch * config['data']['batch_size'], config)
         tf.compat.v1.summary.scalar('learning_rate', learning_rate)
         optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
         if hvd:
            optimizer = hvd.DistributedOptimizer(optimizer)
         train_op = optimizer.minimize(loss, global_step=batch)
         # grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
         # train = optimizer.apply_gradients(grads_and_vars)

         # Add ops to save and restore all the variables.
         saver = tf.train.Saver()

         merged = tf.compat.v1.summary.merge_all()

      logger.info('create session')

      config_proto = tf.compat.v1.ConfigProto()
      if 'gpu' in device_str:
         config_proto.gpu_options.allow_growth = True
         config_proto.gpu_options.visible_device_list = os.environ['CUDA_VISIBLE_DEVICES']
      else:
         config_proto.allow_soft_placement = True
         config_proto.intra_op_parallelism_threads = args.intraop
         config_proto.inter_op_parallelism_threads = args.interop

      # Initialize an iterator over a dataset with 10 elements.
      sess = tf.compat.v1.Session(config=config_proto)

      if rank == 0:
         train_writer = tf.compat.v1.summary.FileWriter(os.path.join(args.logdir, 'train'), sess.graph)
         valid_writer = tf.compat.v1.summary.FileWriter(os.path.join(args.logdir, 'valid'), sess.graph)

      #    train_handle = sess.run(iter_train.string_handle())
      if args.restore:
         logger.info('restoring model: %s', args.restore)
         saver.restore(sess, args.restore)
      else:
         init = tf.compat.v1.global_variables_initializer()
         sess.run(init, {is_training_pl: True})
      if hvd:
         sess.run(hvd.broadcast_global_variables(0))

      logger.info('running over data')
      status_interval = config['training']['status']
      total_acc = 0.
      loss_sum = 0.
      for epoch in range(config['training']['epochs']):
         logger.info('epoch %s of %s  (logdir: %s)', epoch + 1, config['training']['epochs'], args.logdir)
         sess.run(training_init_op)

         start = time.time()
         while True:

            try:
               feed_dict = {is_training_pl: True}
               summary, step, _, loss_val, accuracy_val = sess.run([merged, batch,
                                                                    train_op, loss, accuracy], feed_dict=feed_dict)

               if rank == 0:
                  train_writer.add_summary(summary, step)

               total_acc += accuracy_val
               loss_sum += loss_val

               # logger.info(f'pred_val.shape   = {pred_val.shape}')
               # logger.info(f'target_val.shape = {target_val.shape}')

               if step % status_interval == 0:
                  end = time.time()
                  duration = end - start
                  logger.info('step: %10d   mean loss: %10.6f   accuracy:  %10.6f  imgs/sec: %10.6f',
                              step,
                              loss_sum / float(status_interval), total_acc / float(status_interval),
                              float(status_interval) * config['data']['batch_size'] / duration)
                  start = time.time()
                  total_acc = 0.
                  loss_sum = 0.
            except tf.errors.OutOfRangeError:
               ss = saver.save(sess, os.path.join(args.logdir, "model.ckpt"), global_step=step)

               logger.info(' end of epoch: %s', ss)
               break

         sess.run(validation_init_op)
         logger.info('running validation')
         total_acc = 0.
         total_loss = 0.
         steps = 0.
         while True:
            try:
               feed_dict = {is_training_pl: False}
               summary, valid_step, loss_val, accuracy_val = sess.run([merged, batch, loss, accuracy],
                                                                      feed_dict=feed_dict)
               total_acc += accuracy_val
               total_loss += loss_val
               steps += 1.
               logger.info('valid step: %s', valid_step)
               if rank == 0:
                  valid_writer.add_summary(summary, valid_step)
            except tf.errors.OutOfRangeError:
               total_loss = total_loss / steps
               total_acc = total_acc / steps
               logger.info(' end of validation  mean loss: %10.6f  mean acc: %10.6f', total_loss, total_acc)

               break


if __name__ == "__main__":
   main()
