import tensorflow as tf
import time,logging,glob
import numpy as np
import h5py
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
   
   hf = h5py.File(path,'r')
   images = hf['raw']
   images = np.float32(images)
   labels = hf['truth']
   labels = np.int32(labels)

   shuffle_in_unison(images,labels)
   

   # could do some preprocessing here
   return (images,labels)


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
