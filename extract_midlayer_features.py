"""
Extract midlayer features and save them as numpy npz files
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import GPUtil
from joblib import Parallel, delayed

from optparse import OptionParser
from itertools import *


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

import project_config

model_configure_dict = {}
model_configure_dict['vgg_16_2016_08_28'] = {
    'model_filename': os.path.join(project_config.model_repo_dir, 'vgg_16_2016_08_28/vgg_16.ckpt'),
    'layers_to_extract_list': ['fc6', 'fc7', ],
}


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_cifar10():
    """
    train_images, train_labels, test_images, test_labels = load_cifar10()
    load cifar-10 dataset as numpy array
    :return: train_images, train_labels, test_images, test_labels. images are of shape [number_of_images, height, width, colors]. Images are uint8. labels are unit8
    """
    _IMAGE_SIZE = 32
    _IMAGE_COLOR_CHANNEL = 3
    cifar10_data_repo_dir = os.path.join(project_config.data_repo_dir, 'cifar10/cifar-10-batches-bin/')
    
    train_images = None
    test_images = None
    train_labels = None
    test_labels = None
    # load train
    for train_bin_batch_count in range(5):
        data_bin_filename = os.path.join(cifar10_data_repo_dir, 'data_batch_%d.bin' % (train_bin_batch_count + 1))
        with open(data_bin_filename, 'rb') as fid:
            all_byte = np.fromfile(fid, dtype=np.uint8)
            one_record_len = _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_COLOR_CHANNEL + 1
            all_byte = all_byte.reshape((-1, one_record_len,))
            labels = all_byte[:, 0]
            num_images = all_byte.shape[0]
            images = all_byte[:, 1:].reshape((num_images, 3, 32, 32))
            print('load from %s, num_images=%d' % (data_bin_filename, num_images))
            images = np.transpose(images, [0, 2, 3, 1, ])
            if train_images is None:
                train_images = images
                train_labels = labels
            else:
                train_images = np.vstack([train_images, images])
                train_labels = np.concatenate([train_labels, labels])
        pass  # end with
    pass  # end for train_bin_batch_count
    
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


pass  # end def

FLAGS_batch_size = 128
FLAGS_num_preprocessing_threads = 2


def extract_vgg_16_2016_08_28(options, parameters):
    split_id, num_total_splits = parameters
    
    # num_of_total_gpus = 4
    #
    # time.sleep( 20* (split_id %  options.n_jobs) + 1)
    # time.sleep(np.random.randint(1,1+num_of_total_gpus))
    
    model_name = 'vgg_16_2016_08_28'
    checkpoint_file = model_configure_dict[model_name]['model_filename']
    image_size = vgg.vgg_16.default_image_size
    
    trainset_output_pool5_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/trainset_feat_x%d_sp%d_pool5.npz' % (options.num_perturb, split_id))
    testset_output_pool5_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/testset_feat_x%d_sp%d_pool5.npz' % (options.num_perturb, split_id))
    
    trainset_output_fc6_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/trainset_feat_x%d_sp%d_fc6.npz' % (options.num_perturb, split_id))
    testset_output_fc6_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/testset_feat_x%d_sp%d_fc6.npz' % (options.num_perturb, split_id))
    
    trainset_output_fc7_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/trainset_feat_x%d_sp%d_fc7.npz' % (options.num_perturb, split_id))
    testset_output_fc7_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/testset_feat_x%d_sp%d_fc7.npz' % (options.num_perturb, split_id))
    
    bool_should_run_trainset = True
    bool_should_run_testset = True
    if os.path.isfile(trainset_output_pool5_filename) and os.path.isfile(trainset_output_fc6_filename) and os.path.isfile(trainset_output_fc7_filename):
        bool_should_run_trainset = False
    if os.path.isfile(testset_output_pool5_filename) and os.path.isfile(testset_output_fc6_filename) and os.path.isfile(testset_output_fc7_filename):
        bool_should_run_testset = False
    
    is_training = True if options.num_perturb > 1 else False

    # try:
    #     deviceID_list = GPUtil.getFirstAvailable(order='first', maxLoad=0.5, maxMemory=0.5, attempts=num_of_total_gpus,
    #                                              interval=10 + np.random.randint(0, num_of_total_gpus * 2), verbose=True)
    #     print('!!! GPU found in split %d !!!' % split_id)
    #     # os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceID_list[0])
    #     gpu_device_config = '/gpu:0'
    #     cpu_device_config = '/cpu:%d' % deviceID_list[0] + 1
    # except:
    #     # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #     # num_free_cpu =  (options.num_total_cpu - num_of_total_gpus - 1) # number of free avaiable cpus, exclude cpu:0
    #     cpu_id = num_of_total_gpus - 1 + 1 + 1
    #     gpu_device_config = '/cpu:%d' % cpu_id
    #     cpu_device_config = '/cpu:%d' % cpu_id
    #     print('no GPU found in split %d' % split_id)
    
    if options.use_cpu=='True':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpu_device_config = '/cpu:0'
        cpu_device_config = '/cpu:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(split_id)
        gpu_device_config = '/gpu:0'
        cpu_device_config = '/cpu:%d' % (split_id + 1)
    pass # end if options.use_cpu
    

    with tf.Graph().as_default(), tf.device(cpu_device_config):
        # image_input is a uint8 image, shape=[height, width, color]
        image_input = tf.placeholder(tf.uint8, shape=[32, 32, 3], name='image_input')
        processed_image = vgg_preprocessing.preprocess_image(image_input, image_size, image_size, is_training=is_training)
        processed_images = tf.expand_dims(processed_image, 0)
        
        with slim.arg_scope(vgg.vgg_arg_scope()):
            with tf.device(gpu_device_config): # since we mask GPU via $CUDA_VISIBLE_DEVICES, tf can only see '0' gpu now
                logits, end_points = vgg.vgg_16(processed_images, num_classes=1000, is_training=is_training, dropout_keep_prob=0.1, )
            pass  # end with tf.device
        pass  # end with slim.arg_scope
        
        
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, slim.get_model_variables('vgg_16'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,)) as sess:
            
            #   Load weights
            init_fn(sess)
            
            vgg_16_pool5_layer = end_points['vgg_16/pool5']
            vgg_16_fc6_layer = end_points['vgg_16/fc6']
            vgg_16_fc7_layer = end_points['vgg_16/fc7']
            
            train_images, train_labels, test_images, test_labels = load_cifar10()
            
            if options.debug=='True':
                train_images = train_images[0:500,:]
                train_labels = train_labels[0:500]
                test_images = test_images[0:500,:]
                test_labels = test_labels[0:500]
            pass # end if
            
            train_index_list = np.array_split(range(train_images.shape[0]), num_total_splits)
            train_subsplit_index = train_index_list[split_id]
            train_images = train_images[train_subsplit_index,:]
            train_labels = train_labels[train_subsplit_index]
            
            test_index_list = np.array_split(range(test_images.shape[0]), num_total_splits)
            test_subsplit_index = test_index_list[split_id]
            test_images = test_images[test_subsplit_index, :]
            test_labels = test_labels[test_subsplit_index]
            
            print('split=%d/%d, trainset size=%d, testset size=%d' % (split_id, num_total_splits, train_images.shape[0], test_images.shape[0]))
            
            # extract trainset image midlayer features
            
            if bool_should_run_trainset:
                distutils.dir_util.mkpath(os.path.dirname(trainset_output_fc7_filename))
                n = train_images.shape[0]

                perturb_trainset_pool5_feature_matrix = None
                perturb_trainset_fc6_feature_matrix = None
                perturb_trainset_fc7_feature_matrix = None
                perturb_train_labels = None
                
                
                for perturb_count in range(options.num_perturb):
                    trainset_pool5_feature_matrix = None
                    trainset_fc6_feature_matrix = None
                    trainset_fc7_feature_matrix = None
                    
                    for image_count in range(train_images.shape[0]):
                        vgg_16_pool5_output, vgg_16_fc6_output, vgg_16_fc7_output = \
                            sess.run([vgg_16_pool5_layer, vgg_16_fc6_layer, vgg_16_fc7_layer], feed_dict={
                                image_input: np.squeeze(train_images[image_count,:,:,:]),})
                        
                        if trainset_pool5_feature_matrix is None:
                            trainset_pool5_feature_matrix = np.zeros((n, np.prod(vgg_16_pool5_output.shape[1:])))

                        if trainset_fc6_feature_matrix is None:
                            trainset_fc6_feature_matrix = np.zeros((n, vgg_16_fc6_output.shape[3]))
                            
                        if trainset_fc7_feature_matrix is None:
                            trainset_fc7_feature_matrix = np.zeros((n, vgg_16_fc7_output.shape[3]))

                        trainset_pool5_feature_matrix[image_count, :] = np.ravel(vgg_16_pool5_output)
                        trainset_fc6_feature_matrix[image_count, :] = np.ravel(vgg_16_fc6_output)
                        trainset_fc7_feature_matrix[image_count, :] = np.ravel(vgg_16_fc7_output)

                        if image_count % (n/100)==0:
                            print('[%s] extract trainting split_id=%d, image_count=%d, n=%d, perturb_count=%d' % (gpu_device_config, split_id, image_count, n, perturb_count))
                    pass # end for
                
                    if perturb_count==0:
                        perturb_trainset_pool5_feature_matrix = trainset_pool5_feature_matrix
                        perturb_trainset_fc6_feature_matrix = trainset_fc6_feature_matrix
                        perturb_trainset_fc7_feature_matrix = trainset_fc7_feature_matrix
                        perturb_train_labels = train_labels
                    else:
                        perturb_trainset_pool5_feature_matrix = np.vstack([perturb_trainset_pool5_feature_matrix,trainset_pool5_feature_matrix] )
                        perturb_trainset_fc6_feature_matrix = np.vstack([perturb_trainset_fc6_feature_matrix, trainset_fc6_feature_matrix])
                        perturb_trainset_fc7_feature_matrix = np.vstack([perturb_trainset_fc7_feature_matrix, trainset_fc7_feature_matrix])
                        perturb_train_labels = np.concatenate([perturb_train_labels, train_labels])
                    pass # end if
                    
                pass # end for perturb_count
                # export to numpy files
                if not os.path.isfile(trainset_output_pool5_filename):
                    np.savez_compressed(trainset_output_pool5_filename, features=perturb_trainset_pool5_feature_matrix, labels=perturb_train_labels)

                if not os.path.isfile(trainset_output_fc6_filename):
                    np.savez_compressed(trainset_output_fc6_filename, features=perturb_trainset_fc6_feature_matrix, labels=perturb_train_labels)

                if not os.path.isfile(trainset_output_fc7_filename):
                    np.savez_compressed(trainset_output_fc7_filename, features=perturb_trainset_fc7_feature_matrix, labels=perturb_train_labels)
            pass  # end if

            if bool_should_run_testset:
                distutils.dir_util.mkpath(os.path.dirname(testset_output_fc7_filename))
                n = test_images.shape[0]
    
                perturb_testset_pool5_feature_matrix = None
                perturb_testset_fc6_feature_matrix = None
                perturb_testset_fc7_feature_matrix = None
                perturb_test_labels = None
    
                for perturb_count in range(options.num_perturb):
                    testset_pool5_feature_matrix = None
                    testset_fc6_feature_matrix = None
                    testset_fc7_feature_matrix = None
        
                    for image_count in range(test_images.shape[0]):
                        vgg_16_pool5_output, vgg_16_fc6_output, vgg_16_fc7_output = \
                            sess.run([vgg_16_pool5_layer, vgg_16_fc6_layer, vgg_16_fc7_layer], feed_dict={
                                image_input: np.squeeze(test_images[image_count, :, :, :]), })
            
                        if testset_pool5_feature_matrix is None:
                            testset_pool5_feature_matrix = np.zeros((n, np.prod(vgg_16_pool5_output.shape[1:])))
            
                        if testset_fc6_feature_matrix is None:
                            testset_fc6_feature_matrix = np.zeros((n, vgg_16_fc6_output.shape[3]))
            
                        if testset_fc7_feature_matrix is None:
                            testset_fc7_feature_matrix = np.zeros((n, vgg_16_fc7_output.shape[3]))
            
                        testset_pool5_feature_matrix[image_count, :] = np.ravel(vgg_16_pool5_output)
                        testset_fc6_feature_matrix[image_count, :] = np.ravel(vgg_16_fc6_output)
                        testset_fc7_feature_matrix[image_count, :] = np.ravel(vgg_16_fc7_output)
                        
                        if image_count % (n/100)==0:
                            print('[%s] extract testing split_id=%d, image_count=%d, n=%d, perturb_count=%d' % (gpu_device_config, split_id, image_count, n, perturb_count))
                        
                    pass  # end for
        
                    if perturb_count == 0:
                        perturb_testset_pool5_feature_matrix = testset_pool5_feature_matrix
                        perturb_testset_fc6_feature_matrix = testset_fc6_feature_matrix
                        perturb_testset_fc7_feature_matrix = testset_fc7_feature_matrix
                        perturb_test_labels = test_labels
                    else:
                        perturb_testset_pool5_feature_matrix = np.vstack([perturb_testset_pool5_feature_matrix, testset_pool5_feature_matrix])
                        perturb_testset_fc6_feature_matrix = np.vstack([perturb_testset_fc6_feature_matrix, testset_fc6_feature_matrix])
                        perturb_testset_fc7_feature_matrix = np.vstack([perturb_testset_fc7_feature_matrix, testset_fc7_feature_matrix])
                        perturb_test_labels = np.concatenate([perturb_test_labels, test_labels])
                    pass  # end if
    
                pass  # end for perturb_count
                # export to numpy files
                if not os.path.isfile(testset_output_pool5_filename):
                    np.savez_compressed(testset_output_pool5_filename, features=perturb_testset_pool5_feature_matrix, labels=perturb_test_labels)
    
                if not os.path.isfile(testset_output_fc6_filename):
                    np.savez_compressed(testset_output_fc6_filename, features=perturb_testset_fc6_feature_matrix, labels=perturb_test_labels)
    
                if not os.path.isfile(testset_output_fc7_filename):
                    np.savez_compressed(testset_output_fc7_filename, features=perturb_testset_fc7_feature_matrix, labels=perturb_test_labels)
            pass  # end if bool_should_run_testset
        
        pass # end tf.Session
    
    pass  # end with tf.Graph().as_default():
pass  # end def main

if __name__ == '__main__':
    """
    python extract_midlayer_features.py --n_jobs=8 --num_total_splits=20 --num_perturb=2 --debug=True
    """
    parser = OptionParser()
    parser.add_option('--use_cpu', type='string', dest='use_cpu', default='False', help='Only use CPU.')
    parser.add_option('--n_jobs', type='int', dest='n_jobs', default=1, help='number of parallel jobs.')
    parser.add_option('-t', '--num_total_splits', type='int', dest='num_total_splits', default=1, help='number of total splits.')
    parser.add_option('-p', '--num_perturb', type='int', dest='num_perturb', default=1, help='number of random perturbation for each image.')
    parser.add_option('--debug', type='string', dest='debug', default=False, help='run debug code.')
    (options, args) = parser.parse_args()
    
    if options.use_cpu=='False' and (options.n_jobs!=options.num_total_splits):
        print('when using GPU, n_jobs must equal to num_total_splits')
        exit(1)
        
    # generate task list
    task_list = []
    for split_id in range(options.num_total_splits):
        task_list.append([split_id, options.num_total_splits])
    pass # end for

    par_results = Parallel(n_jobs=options.n_jobs, verbose=50, batch_size=1)(delayed(extract_vgg_16_2016_08_28)(options, par_for_parameters) for par_for_parameters in task_list)