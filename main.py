#!/usr/bin/env python
import argparse,logging,json,time,os
import tensorflow as tf
import data_handler
import numpy as np
from models import pointnet_seg
import horovod
import horovod.tensorflow as hvd
hvd.init()
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = 'config.json'
DEFAULT_INTEROP = 1
DEFAULT_INTRAOP = os.cpu_count()
DEFAULT_LOGDIR = '/tmp/tf-'+str(os.getpid())

tf.debugging.set_log_device_placement(True)

def main():
   ''' simple starter program for tensorflow models. '''
   logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:' + ('%05d' % hvd.rank()) + ':%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config',dest='config_filename',help='configuration filename in json format [default: %s]' % DEFAULT_CONFIG,default=DEFAULT_CONFIG)
   parser.add_argument('--interop',help='set Tensorflow "inter_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTEROP,default=DEFAULT_INTEROP)
   parser.add_argument('--intraop',help='set Tensorflow "intra_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTRAOP,default=DEFAULT_INTRAOP)
   parser.add_argument('-l','--logdir',default=DEFAULT_LOGDIR,help='define location to save log information [default: %s]' % DEFAULT_LOGDIR)

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   logging.info('using tensorflow version:   %s',tf.__version__)
   logging.info('using tensorflow from:      %s',tf.__file__)
   logging.info('using horovod version:      %s',horovod.__version__)
   logging.info('using horovod from:         %s',horovod.__file__)
   logging.info('logdir:                     %s',args.logdir)
   logging.info('interop:                    %s',args.interop)
   logging.info('intraop:                    %s',args.intraop)
   
   device_str = '/cpu:0'
   if tf.test.is_gpu_available():
      device_str = '/gpu:' + str(hvd.local_rank())
   logger.info('device:                     %s',device_str)

   config = json.load(open(args.config_filename))
   config['device'] = device_str
   
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   logger.info('%s = \n %s',args.config_filename,json.dumps(config,indent=4,sort_keys=True))
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')

   with tf.Graph().as_default():

      logger.info('getting datasets')
      trainds,validds = data_handler.get_datasets(config)

      iterator = tf.compat.v1.data.Iterator.from_structure((tf.float32,tf.int32),((config['data']['batch_size'],config['data']['num_points'],config['data']['num_features']),(config['data']['batch_size'],config['data']['num_points'])))
      input,target = iterator.get_next()
      training_init_op = iterator.make_initializer(trainds)

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

         nclasses = len(config['data']['classes'])

         pred,endpoints = pointnet_seg.get_model(input,is_training_pl,nclasses)
         loss = pointnet_seg.get_loss(pred,target,endpoints)
         tf.compat.v1.summary.scalar('loss',loss)

         accuracy = pointnet_seg.get_accuracy(pred,target)
         tf.compat.v1.summary.scalar('accuracy', accuracy)

         learning_rate = pointnet_seg.get_learning_rate(batch,config) * hvd.size()
         tf.compat.v1.summary.scalar('learning_rate',learning_rate)
         optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
         optimizer = hvd.DistributedOptimizer(optimizer)
         train_op = optimizer.minimize(loss, global_step=batch)
         #grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
         #train = optimizer.apply_gradients(grads_and_vars)

         # Add ops to save and restore all the variables.
         # saver = tf.train.Saver()

         merged = tf.compat.v1.summary.merge_all()

      logger.info('create session')

      config_proto = tf.compat.v1.ConfigProto()
      config_proto.allow_soft_placement = True
      config_proto.intra_op_parallelism_threads = args.intraop
      config_proto.inter_op_parallelism_threads = args.interop
      if 'gpu' in device_str:
         config_proto.gpu_options.allow_growth = True
         config_proto.gpu_options.visible_device_list = str(hvd.local_rank())

      # Initialize an iterator over a dataset with 10 elements.
      sess = tf.compat.v1.Session(config=config_proto)
      logger.info('initialize dataset iterator')

      train_writer = tf.compat.v1.summary.FileWriter(os.path.join(args.logdir, 'train'),sess.graph)
      # test_writer = tf.summary.FileWriter(os.path.join(args.logdir, 'test'))
      with tf.device(device_str):
         sess.run(training_init_op)
      #    train_handle = sess.run(iter_train.string_handle())
      
      init = tf.compat.v1.global_variables_initializer()
      sess.run(init, {is_training_pl: True})
      sess.run(hvd.broadcast_global_variables(0))

      logger.info('running over data')
      is_training = True
      status_interval = config['training']['status']
      total_acc = 0.
      loss_sum = 0.
      for epoch in range(config['training']['epochs']):
         logger.info('epoch %s of %s  (logdir: %s)',epoch + 1,config['training']['epochs'],args.logdir)
         
         while True:
            try:
               feed_dict = {is_training_pl: is_training}
               summary, step, _, loss_val, accuracy_val = sess.run([merged, batch,
                   train_op, loss, accuracy], feed_dict=feed_dict)
               train_writer.add_summary(summary, step)
               
               total_acc += accuracy_val
               loss_sum += loss_val

               # logger.info(f'pred_val.shape   = {pred_val.shape}')
               # logger.info(f'target_val.shape = {target_val.shape}')
               
               if step % status_interval == 0:
                  logger.info('mean loss: %10.6f   accuracy:  %10.6f',loss_sum / float(status_interval),total_acc / float(status_interval))

                  total_acc = 0.
                  loss_sum = 0.
            except tf.errors.OutOfRangeError:
               logger.info(' end of epoch ')
               break




if __name__ == "__main__":
   main()
