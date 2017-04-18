"""
Extract midlayer features and save them as numpy npz files
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def extract_vgg_16_2016_08_28(options):
    model_name = 'vgg_16_2016_08_28'
    checkpoint_file = model_configure_dict[model_name]['model_filename']
    image_size = vgg.vgg_16.default_image_size
    
    trainset_output_pool5_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/trainset_feat_x%d_pool5.npz' % options.num_perturb)
    testset_output_pool5_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/testset_feat_x%d_pool5.npz' % options.num_perturb)
    
    trainset_output_fc6_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/trainset_feat_x%d_fc6.npz' % options.num_perturb)
    testset_output_fc6_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/testset_feat_x%d_fc6.npz' % options.num_perturb)
    
    trainset_output_fc7_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/trainset_feat_x%d_fc7.npz' % options.num_perturb)
    testset_output_fc7_filename = os.path.join(project_config.output_dir, 'midlayer_feat/cifar10/vgg_16/testset_feat_x%d_fc7.npz' % options.num_perturb)
    
    bool_should_run_trainset = True
    bool_should_run_testset = True
    if os.path.isfile(trainset_output_pool5_filename) and os.path.isfile(trainset_output_fc6_filename) and os.path.isfile(trainset_output_fc7_filename):
        bool_should_run_trainset = False
    if os.path.isfile(testset_output_pool5_filename) and os.path.isfile(testset_output_fc6_filename) and os.path.isfile(testset_output_fc7_filename):
        bool_should_run_testset = False
    
    is_training = True if options.num_perturb > 1 else False
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        
        
        # processed_image = [None, ] * options.num_gpu
        # processed_images = [None, ] * options.num_gpu
        
        # sess = tf.InteractiveSession()

        if options.use_cpu == 'True':
            end_points = [None, ]
            image_input = [None, ]
            gpu_count = 0
            image_input[gpu_count] = tf.placeholder(tf.uint8, shape=[32, 32, 3], name='image_input_%d' % (gpu_count))
            processed_image = vgg_preprocessing.preprocess_image(image_input[gpu_count], image_size, image_size, is_training=is_training)
            processed_images = tf.expand_dims(processed_image, 0)
            with slim.arg_scope(vgg.vgg_arg_scope()):
                with tf.device('/cpu:%d' % (gpu_count)):
                    logits, tmp_end_points = vgg.vgg_16(processed_images, num_classes=1000, is_training=is_training, dropout_keep_prob=0.1, )
                    end_points[gpu_count] = tmp_end_points
                pass  # end with tf.device
            pass  # end with slim.arg_scope
        else:
            end_points = [None, ] * options.num_gpu
            image_input = [None, ] * options.num_gpu
            for gpu_count in range(options.num_gpu):
                # image_input is a uint8 image, shape=[height, width, color]
                image_input[gpu_count] = tf.placeholder(tf.uint8, shape=[32, 32, 3], name='image_input_%d' % (gpu_count))
                processed_image = vgg_preprocessing.preprocess_image(image_input[gpu_count], image_size, image_size, is_training=is_training)
                processed_images = tf.expand_dims(processed_image, 0)
                with slim.arg_scope(vgg.vgg_arg_scope()):
                    with tf.device('/gpu:%d' % (gpu_count)):
                        logits, tmp_end_points = vgg.vgg_16(processed_images, num_classes=1000, is_training=is_training, dropout_keep_prob=0.1, )
                        end_points[gpu_count] = tmp_end_points
                    pass  # end with tf.device
                pass  # end with slim.arg_scope

            pass  # end for gpu_count
        pass # end if options.num_gpu <= 0:
        
        
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, slim.get_model_variables('vgg_16'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            
            #   Load weights
            init_fn(sess)
            
            vgg_16_pool5_layer = [end_points[i]['vgg_16/pool5'] for i in range(options.num_gpu)]
            vgg_16_fc6_layer = [end_points[i]['vgg_16/fc6'] for i in range(options.num_gpu)]
            vgg_16_fc7_layer = [end_points[i]['vgg_16/fc7'] for i in range(options.num_gpu)]
            
            train_images, train_labels, test_images, test_labels = load_cifar10()
            print('trainset size=%d, testset size=%d' % (train_images.shape[0], test_images.shape[0]))
            
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
                    
                    for mini_batch_count, mini_batch_image_index_group in enumerate(grouper(range(n), options.num_gpu)):
                        image_index_list = [i for i in mini_batch_image_index_group if i is not None]
                        mini_batch_size = len(image_index_list)
                        feed_dict = {}
                        for i in range(mini_batch_size):
                            feed_dict[image_input[i]] = np.squeeze(train_images[image_index_list[i], :, :, :])
                        pass # end for i
    
                        sess_graph_to_run_list = []
                        sess_graph_to_run_list += [vgg_16_pool5_layer[i] for i in range(mini_batch_size)]
                        sess_graph_to_run_list += [vgg_16_fc6_layer[i] for i in range(mini_batch_size)]
                        sess_graph_to_run_list += [vgg_16_fc7_layer[i] for i in range(mini_batch_size)]
                        network_output = sess.run(sess_graph_to_run_list,feed_dict=feed_dict)

                        tmp_counter = 0
                        vgg_16_pool5_output_list = network_output[(tmp_counter*mini_batch_size+0):(tmp_counter*mini_batch_size+mini_batch_size)]
                        tmp_counter += 1
                        vgg_16_fc6_output_list = network_output[(tmp_counter * mini_batch_size + 0):(tmp_counter * mini_batch_size + mini_batch_size)]
                        tmp_counter += 1
                        vgg_16_fc7_output_list = network_output[(tmp_counter * mini_batch_size + 0):(tmp_counter * mini_batch_size + mini_batch_size)]
                        tmp_counter += 1

                        if trainset_pool5_feature_matrix is None:
                            trainset_pool5_feature_matrix = np.zeros((n, np.prod(vgg_16_pool5_output_list[0].shape[1:])))

                        if trainset_fc6_feature_matrix is None:
                            trainset_fc6_feature_matrix = np.zeros((n, vgg_16_fc6_output_list[0].shape[3]))

                        if trainset_fc7_feature_matrix is None:
                            trainset_fc7_feature_matrix = np.zeros((n, vgg_16_fc7_output_list[0].shape[3]))

                        for i in range(mini_batch_size):
                            trainset_pool5_feature_matrix[image_index_list[i], :] = np.ravel(vgg_16_pool5_output_list[i][0,])
                            trainset_fc6_feature_matrix[image_index_list[i], :] = vgg_16_fc6_output_list[i][0, 0, 0, :]
                            trainset_fc7_feature_matrix[image_index_list[i], :] = vgg_16_fc7_output_list[i][0, 0, 0, :]
                        pass # end for
                        sys.stdout.write('\r')
                        sys.stdout.write('extract training image batch_count=%d, batch_size=%d, n=%d,  perturb_count=%d' % (mini_batch_count, mini_batch_size, n, perturb_count))
                        sys.stdout.flush()
                    pass # end for mini_batch_image_index_group
                
                    if perturb_count==0:
                        perturb_trainset_pool5_feature_matrix = trainset_pool5_feature_matrix
                        perturb_trainset_fc6_feature_matrix = trainset_fc6_feature_matrix
                        perturb_trainset_fc7_feature_matrix = trainset_fc7_feature_matrix
                        perturb_train_labels = train_labels
                    else:
                        perturb_trainset_pool5_feature_matrix = np.vstack([perturb_trainset_pool5_feature_matrix,trainset_pool5_feature_matrix] )
                        perturb_trainset_fc6_feature_matrix = np.vstack([perturb_trainset_fc6_feature_matrix, trainset_fc6_feature_matrix])
                        perturb_trainset_fc7_feature_matrix = np.vstack([perturb_trainset_fc7_feature_matrix, trainset_fc7_feature_matrix])
                        perturb_train_labels = np.concatenate(perturb_train_labels, train_labels)
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

            # extract testset image midlayer features

            if bool_should_run_testset:
                distutils.dir_util.mkpath(os.path.dirname(testset_output_fc7_filename))
                n = test_images.shape[0]
                n_after_perturb = n * options.num_perturb
    
                perturb_testset_pool5_feature_matrix = None
                perturb_testset_fc6_feature_matrix = None
                perturb_testset_fc7_feature_matrix = None
                perturb_test_labels = None
    
                for perturb_count in range(options.num_perturb):
                    testset_pool5_feature_matrix = None
                    testset_fc6_feature_matrix = None
                    testset_fc7_feature_matrix = None
        
                    for mini_batch_count, mini_batch_image_index_group in enumerate(grouper(range(n), options.num_gpu)):
                        image_index_list = [i for i in mini_batch_image_index_group if i is not None]
                        mini_batch_size = len(image_index_list)
                        feed_dict = {}
                        for i in range(mini_batch_size):
                            feed_dict[image_input[i]] = np.squeeze(test_images[image_index_list[i], :, :, :])
                        pass  # end for i
            
                        sess_graph_to_run_list = []
                        sess_graph_to_run_list += [vgg_16_pool5_layer[i] for i in range(mini_batch_size)]
                        sess_graph_to_run_list += [vgg_16_fc6_layer[i] for i in range(mini_batch_size)]
                        sess_graph_to_run_list += [vgg_16_fc7_layer[i] for i in range(mini_batch_size)]
                        network_output = sess.run(sess_graph_to_run_list, feed_dict=feed_dict)
            
                        tmp_counter = 0
                        vgg_16_pool5_output_list = network_output[(tmp_counter * mini_batch_size + 0):(tmp_counter * mini_batch_size + mini_batch_size)]
                        tmp_counter += 1
                        vgg_16_fc6_output_list = network_output[(tmp_counter * mini_batch_size + 0):(tmp_counter * mini_batch_size + mini_batch_size)]
                        tmp_counter += 1
                        vgg_16_fc7_output_list = network_output[(tmp_counter * mini_batch_size + 0):(tmp_counter * mini_batch_size + mini_batch_size)]
                        tmp_counter += 1
            
                        if testset_pool5_feature_matrix is None:
                            testset_pool5_feature_matrix = np.zeros((n, np.prod(vgg_16_pool5_output_list[0].shape[1:])))
            
                        if testset_fc6_feature_matrix is None:
                            testset_fc6_feature_matrix = np.zeros((n, vgg_16_fc6_output_list[0].shape[3]))
            
                        if testset_fc7_feature_matrix is None:
                            testset_fc7_feature_matrix = np.zeros((n, vgg_16_fc7_output_list[0].shape[3]))

                        for i in range(mini_batch_size):
                            testset_pool5_feature_matrix[image_index_list[i], :] = np.ravel(vgg_16_pool5_output_list[i][0])
                            testset_fc6_feature_matrix[image_index_list[i], :] = vgg_16_fc6_output_list[i][0, 0, 0, :]
                            testset_fc7_feature_matrix[image_index_list[i], :] = vgg_16_fc7_output_list[i][0, 0, 0, :]
                        pass  # end for
                        sys.stdout.write('\r')
                        sys.stdout.write('extract testing image minibatch %d, n=%d,  perturb_count=%d' % (mini_batch_count, n, perturb_count))
                        sys.stdout.flush()
                    pass  # end for mini_batch_image_index_group
        
                    if perturb_count == 0:
                        perturb_testset_pool5_feature_matrix = testset_pool5_feature_matrix
                        perturb_testset_fc6_feature_matrix = testset_fc6_feature_matrix
                        perturb_testset_fc7_feature_matrix = testset_fc7_feature_matrix
                        perturb_test_labels = test_labels
                    else:
                        perturb_testset_pool5_feature_matrix = np.vstack([perturb_testset_pool5_feature_matrix, testset_pool5_feature_matrix])
                        perturb_testset_fc6_feature_matrix = np.vstack([perturb_testset_fc6_feature_matrix, testset_fc6_feature_matrix])
                        perturb_testset_fc7_feature_matrix = np.vstack([perturb_testset_fc7_feature_matrix, testset_fc7_feature_matrix])
                        perturb_test_labels = np.concatenate(perturb_test_labels, test_labels)
                    pass  # end if
    
                pass  # end for perturb_count
                # export to numpy files
                if not os.path.isfile(testset_output_pool5_filename):
                    np.savez_compressed(testset_output_pool5_filename, features=perturb_testset_pool5_feature_matrix, labels=perturb_test_labels)
    
                if not os.path.isfile(testset_output_fc6_filename):
                    np.savez_compressed(testset_output_fc6_filename, features=perturb_testset_fc6_feature_matrix, labels=perturb_test_labels)
    
                if not os.path.isfile(testset_output_fc7_filename):
                    np.savez_compressed(testset_output_fc7_filename, features=perturb_testset_fc7_feature_matrix, labels=perturb_test_labels)
            pass  # end if
        pass # end tf.Session
    pass  # end with tf.Graph().as_default():
pass  # end def main

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-g', '--num_gpu', type='int', dest='num_gpu', default=1, help='number of GPU to use.')
    parser.add_option('-c', '--use_cpu', type='string', dest='use_cpu', default='False', help='Only use CPU.')
    parser.add_option('-p', '--num_perturb', type='int', dest='num_perturb', default=1, help='number of random perturbation for each image.')
    (options, args) = parser.parse_args()
    extract_vgg_16_2016_08_28(options)