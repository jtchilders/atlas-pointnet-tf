import tensorflow as tf
import time,logging,glob
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

config = None


def get_datasets(config_file):
   global config
   config = config_file

   if 'train_glob' in config['data'] and 'valid_glob' in config['data']:
      train = simple_dataset_from_glob(config['data']['train_glob'])
      valid = simple_dataset_from_glob(config['data']['valid_glob'])
   elif 'glob' in config_file['data'] and 'train_fraction' in config_file['data']:
      train,valid = split_filelists(config['data']['glob'],config['data']['train_fraction'])
      train = simple_dataset_from_filelist(train)
      valid = simple_dataset_from_filelist(valid)
   else:
      raise Exception('data set reading not configured correctly')

   return train,valid


def simple_dataset_from_glob(glob_string):

   # glob for the input files
   filelist = tf.data.Dataset.list_files(glob_string)
   # shuffle and repeat at the input file level
   filelist = filelist.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000,count=config['training']['epochs']))
   # map to read files in parallel
   ds = filelist.map(load_file_and_preprocess,num_parallel_calls=config['data']['num_parallel_readers'])
   
   # flatten the inputs across file boundaries
   ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
   
   # speficy batch size
   ds = ds.batch(config['data']['batch_size'])
   
   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=config['data']['prefectch_buffer_size'])
   
   return ds


def simple_dataset_from_filelist(filelist):

   # glob for the input files
   filelist = tf.data.Dataset.from_tensor_slices(filelist)
   # shuffle and repeat at the input file level
   filelist = filelist.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000,count=config['training']['epochs']))
   # map to read files in parallel
   ds = filelist.map(load_file_and_preprocess,num_parallel_calls=config['data']['num_parallel_readers'])
   
   # flatten the inputs across file boundaries
   ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
   
   # speficy batch size
   ds = ds.batch(config['data']['batch_size'])
   
   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=config['data']['prefectch_buffer_size'])
   
   return ds


def split_filelists(glob_str,train_fraction):

   logger.info('using glob string to find input files: %s',glob_str)

   # glob for full filelist
   full_filelist = sorted(glob.glob(glob_str))
   nfiles = len(full_filelist)

   ntrain = int(nfiles * train_fraction)

   train_filelist = full_filelist[:ntrain]
   valid_filelist = full_filelist[ntrain:]

   return train_filelist,valid_filelist


def load_file_and_preprocess(path):
   pyf = tf.py_func(wrapped_loader,[path],(tf.float32,tf.int32))
   return pyf


#def load_file_and_preprocess(path):
def wrapped_loader(path):
   path = path.decode('utf-8')
   col_names    = ['id', 'index', 'x', 'y', 'z', 'r', 'eta', 'phi', 'Et','pid','n','trk_good','trk_id','trk_pt']
   col_dtype    = {'id': np.int64, 'index': np.int32, 'x': np.float32, 'y': np.float32,
                   'z': np.float32, 'eta': np.float32, 'phi': np.float32, 'r': np.float32,
                   'Et': np.float32, 'pid': np.float32, 'n': np.float32,
                   'trk_good': np.float32, 'trk_id': np.float32, 'trk_pt': np.float32}

   data = pd.read_csv(path,header=None,names=col_names, dtype=col_dtype, sep='\t')
   
   # could do some preprocessing here
   return (get_input(data),get_labels(data))


def get_input(data):
   input = data[['r','eta','phi','Et']].to_numpy()

   min = input.min(0)
   # logger.info('min = %s',min)
   max = input.max(0)
   # logger.info('max = %s',max)
   input = (input - min) / (max - min)

   input = np.float32(input)

   # create a weight vector with 1's where points exist and 0's where they do not
   # weights = np.zeros(self.img_shape[0],dtype=np.float32)
   # weights[0:input.shape[0]] = np.ones(input.shape[0],dtype=np.float32)
   img_shape = (1,config['data']['num_points'],config['data']['num_features'])
   padded_input = np.zeros(img_shape,dtype=np.float32)

   padded_input[0,:input.shape[0],:] = input

   # input = padded_input.transpose()
   
   return padded_input


def get_labels(data):
   target = data['pid']
   target = target.to_numpy()
   target = target >= -90
   target = np.int32(target)
   # logger.info('target  #0 = %s #1 = %s # = %s',(target == 0).sum(),(target == 1).sum(),np.prod(target.shape))
   #img_shape = (config['data']['num_points'])
   padded_target = np.zeros((1,config['data']['num_points']),dtype=np.int32)
   padded_target[0,:target.shape[0]] = target

   return padded_target
