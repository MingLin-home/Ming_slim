"""
Extract midlayer features and save them as numpy npz files
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import distutils.dir_util
import sys

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import vgg
from preprocessing import vgg_preprocessing

slim = tf.contrib.slim

import  project_config


model_configure_dict={}
model_configure_dict['vgg_16_2016_08_28']={
  'model_filename': os.path.join(project_config.model_repo_dir,'vgg_16_2016_08_28/vgg_16.ckpt'),
  'layers_to_extract_list': ['fc6','fc7',],
}

def load_cifar10():
  """
  train_images, train_labels, test_images, test_labels = load_cifar10()
  load cifar-10 dataset as numpy array
  :return: train_images, train_labels, test_images, test_labels. images are of shape [number_of_images, height, width, colors]. Images are uint8. labels are unit8
  """
  _IMAGE_SIZE = 32
  _IMAGE_COLOR_CHANNEL = 3
  cifar10_data_repo_dir = os.path.join(project_config.data_repo_dir,'cifar10/cifar-10-batches-bin/')
  
  train_images = None
  test_images = None
  train_labels = None
  test_labels = None
  # load train
  for train_bin_batch_count in range(5):
    data_bin_filename = os.path.join(cifar10_data_repo_dir,'data_batch_%d.bin' % (train_bin_batch_count+1) )
    with open(data_bin_filename, 'rb') as fid:
      all_byte = np.fromfile(fid, dtype=np.uint8)
      one_record_len = _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_COLOR_CHANNEL + 1
      all_byte = all_byte.reshape((-1, one_record_len,))
      labels = all_byte[:, 0]
      num_images = all_byte.shape[0]
      images = all_byte[:, 1:].reshape((num_images, 3, 32, 32))
      print('load from %s, num_images=%d' % (data_bin_filename, num_images))
      images = np.transpose(images,[0,2,3,1,])
      if train_images is None:
        train_images = images
        train_labels = labels
      else:
        train_images = np.vstack([train_images,images])
        train_labels = np.concatenate([train_labels,labels])
    pass # end with
  pass # end for train_bin_batch_count
  
  # load test
  data_bin_filename = os.path.join(cifar10_data_repo_dir, 'test_batch.bin')
  with open(data_bin_filename, 'rb') as fid:
    all_byte = np.fromfile(fid, dtype=np.uint8)
    one_record_len = _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_COLOR_CHANNEL + 1
    all_byte = all_byte.reshape((-1, one_record_len,))
    labels = all_byte[:, 0]
    num_images = all_byte.shape[0]
    images = all_byte[:, 1:].reshape((num_images, 3, 32, 32))
    print('load from %s, num_images=%d' % (data_bin_filename, num_images))
    images = np.transpose(images, [0, 2, 3, 1, ])
    if test_images is None:
      test_images = images
      test_labels = labels
    else:
      test_images = np.vstack([test_images, images])
      test_labels = np.concatenate([test_labels, labels])
  pass  # end with

  return train_images, train_labels, test_images, test_labels

pass # end def
    
FLAGS_batch_size = 128
FLAGS_num_preprocessing_threads = 2

def main():
  model_name = 'vgg_16_2016_08_28'
  checkpoint_file = model_configure_dict[model_name]['model_filename']
  image_size = vgg.vgg_16.default_image_size
  trainset_output_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/trainset_feat_fc7.npz')
  testset_output_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/testset_feat_fc7.npz')
  
  with tf.Graph().as_default():
    # image_input is a uint8 image, shape=[height, width, color]
    image_input = tf.placeholder(tf.uint8,shape=[32,32,3], name='image_input')
    processed_image = vgg_preprocessing.preprocess_image(image_input, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)
    with tf.device('/gpu:0'):
      with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)
      pass # end with
    pass # end with tf.device

    init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, slim.get_model_variables('vgg_16'))

    # sess = tf.InteractiveSession()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      #   Load weights
      init_fn(sess)
      vgg_16_fc7_layer = end_points['vgg_16/fc7']
      train_images, train_labels, test_images, test_labels = load_cifar10()
      print('trainset size=%d, testset size=%d' %(train_images.shape[0], test_images.shape[0]))

      # extract trainset image midlayer features
      if not os.path.isfile(trainset_output_filename):
        distutils.dir_util.mkpath(os.path.dirname(trainset_output_filename))
        n = train_images.shape[0]
        fc7_feat_dim = 4096
        trainset_fc7_feature_matrix = np.zeros((n,fc7_feat_dim)  )
        for img_count in range(train_images.shape[0]):
          np_image, network_input, vgg_16_fc7_output = sess.run([image_input, processed_image, vgg_16_fc7_layer],
                                                                feed_dict={image_input: np.squeeze(train_images[img_count,:,:,:])})
          trainset_fc7_feature_matrix[img_count,:] = vgg_16_fc7_output[0,0,0,:]
          sys.stdout.write('\r')
          sys.stdout.write('extract training image %d/%d' %(img_count,train_images.shape[0]))
          sys.stdout.flush()
        pass # end for
        # export to numpy files
        if not os.path.isfile(trainset_output_filename):
          np.savez_compressed(trainset_output_filename, features=trainset_fc7_feature_matrix, labels=train_labels)
      pass # end if
      
      # extract testset image midlayer features
      if not os.path.isfile(testset_output_filename):
        distutils.dir_util.mkpath(os.path.dirname(testset_output_filename))
        n = test_images.shape[0]
        fc7_feat_dim = 4096
        testset_fc7_feature_matrix = np.zeros((n, fc7_feat_dim))
        for img_count in range(test_images.shape[0]):
          np_image, network_input, vgg_16_fc7_output = sess.run([image_input, processed_image, vgg_16_fc7_layer],
                                                                feed_dict={image_input: np.squeeze(test_images[img_count, :, :, :])})
          testset_fc7_feature_matrix[img_count, :] = vgg_16_fc7_output[0, 0, 0, :]
          sys.stdout.write('\r')
          sys.stdout.write('extract testing image %d/%d' % (img_count, test_images.shape[0]))
          sys.stdout.flush()
        pass # end for
        if not os.path.isfile(testset_output_filename):
          np.savez_compressed(testset_output_filename, features=testset_fc7_feature_matrix, labels=test_labels)
      pass # end if
  pass # end with tf.Graph().as_default():

  
  
  


pass  # end def main

if __name__ == '__main__':
  main()