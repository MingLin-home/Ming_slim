"""
Extract midlayer features and save them as tfrec tfd files
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
from preprocessing import cifar10_vgg_preprocessing

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

def build_vgg16_network( gpu_device_config, is_training):
    image_size = vgg.vgg_16.default_image_size
    image_input = tf.placeholder(tf.uint8, shape=[32, 32, 3], name='image_input')
    processed_image = cifar10_vgg_preprocessing.preprocess_image(image_input, image_size, image_size, is_training=is_training,
                                                                 )
    processed_images = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        with tf.device(gpu_device_config):  # since we mask GPU via $CUDA_VISIBLE_DEVICES, tf can only see '0' gpu now
            logits, end_points = vgg.vgg_16(processed_images, num_classes=1000, is_training=is_training, dropout_keep_prob=0.5, )
        pass  # end with tf.device
    pass  # end with slim.arg_scope
    return logits, end_points, image_input
pass # end def

def extract_vgg_16_features(end_points, sess, image_input,
        train_images, gpu_device_config, real_gpu_device_config, cpu_device_config, checkpoint_file, perturb_count=-1, is_training=False,
                            ):

    """
    pool5_feature_matrix, fc6_feature_matrix, fc7_feature_matrix =  extract_vgg_16_features(...)
    :return:
    """
    image_size = vgg.vgg_16.default_image_size
    n = train_images.shape[0]
    # vgg_16_pool4_layer = end_points['vgg_16/pool4']
    vgg_16_pool5_layer = end_points['vgg_16/pool5']
    vgg_16_fc6_layer = end_points['vgg_16/fc6']
    vgg_16_fc7_layer = end_points['vgg_16/fc7']

    # trainset_pool4_feature_matrix = None
    trainset_pool5_feature_matrix = None
    trainset_fc6_feature_matrix = None
    trainset_fc7_feature_matrix = None

    for image_count in range(train_images.shape[0]):
        vgg_16_pool5_output, vgg_16_fc6_output, vgg_16_fc7_output = \
            sess.run([vgg_16_pool5_layer, vgg_16_fc6_layer, vgg_16_fc7_layer], feed_dict={
                image_input: np.squeeze(train_images[image_count, :, :, :]), })

        # if trainset_pool4_feature_matrix is None:
        #     trainset_pool4_feature_matrix = np.zeros((n, np.prod(vgg_16_pool4_output.shape[1:])))

        if trainset_pool5_feature_matrix is None:
            trainset_pool5_feature_matrix = np.zeros((n, np.prod(vgg_16_pool5_output.shape[1:])))

        if trainset_fc6_feature_matrix is None:
            trainset_fc6_feature_matrix = np.zeros((n, vgg_16_fc6_output.shape[3]))

        if trainset_fc7_feature_matrix is None:
            trainset_fc7_feature_matrix = np.zeros((n, vgg_16_fc7_output.shape[3]))

        # trainset_pool4_feature_matrix[image_count, :] = np.ravel(vgg_16_pool4_output)
        trainset_pool5_feature_matrix[image_count, :] = np.ravel(vgg_16_pool5_output)
        trainset_fc6_feature_matrix[image_count, :] = np.ravel(vgg_16_fc6_output)
        trainset_fc7_feature_matrix[image_count, :] = np.ravel(vgg_16_fc7_output)

        if image_count % (train_images.shape[0] / 100) == 0:
            print('[%s] image_count=%d, n=%d, perturb_count=%d' % (real_gpu_device_config, image_count, n, perturb_count))
    pass  # end for


    return trainset_pool5_feature_matrix, trainset_fc6_feature_matrix, trainset_fc7_feature_matrix

pass # end def


def extract_vgg_16_2016_08_28(options, parameters):
    num_trainset_blocks = options.num_trainset_blocks
    num_testset_blocks = options.num_testset_blocks
    
    gpu_id, num_gpus = parameters
    model_name = 'vgg_16_2016_08_28'
    checkpoint_file = model_configure_dict[model_name]['model_filename']
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if options.debug=='True':
        gpu_device_config = '/cpu:0'
    else:
        gpu_device_config = '/gpu:0'
    real_gpu_device_config = str(gpu_id)
    cpu_device_config = '/cpu:%d' % (gpu_id + 1)
    

    train_images, train_labels, test_images, test_labels = load_cifar10()
    
    

    if options.debug == 'True':
        train_images = train_images[0:500, :]
        train_labels = train_labels[0:500]
        test_images = test_images[0:500, :]
        test_labels = test_labels[0:500]
    pass  # end if

    train_index_list = np.array_split(range(train_images.shape[0]), num_gpus)
    train_subsplit_index = train_index_list[gpu_id]
    train_images = train_images[train_subsplit_index, :]
    train_labels = train_labels[train_subsplit_index]

    test_index_list = np.array_split(range(test_images.shape[0]), num_gpus)
    test_subsplit_index = test_index_list[gpu_id]
    test_images = test_images[test_subsplit_index, :]
    test_labels = test_labels[test_subsplit_index]

    print('split=%d/%d, trainset size=%d, testset size=%d' % (gpu_id, num_gpus, train_images.shape[0], test_images.shape[0]))

    with tf.Graph().as_default(), tf.device(cpu_device_config):
        # build is_training=False network
        logits, end_points, image_input = build_vgg16_network( gpu_device_config, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, slim.get_model_variables('vgg_16'))
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, ))
        #   Load weights
        init_fn(sess)

        trainset_block_index_list = np.array_split( range(train_images.shape[0]), num_trainset_blocks )

        for trainset_block_count, trainset_block_index in enumerate(trainset_block_index_list):
            for perturb_count in range(1):
                # trainset_output_pool4_filename = os.path.join(
                #     project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/trainset_feat_pert%d_sp%d_bl%d_pool4.tfd' % (perturb_count, gpu_id, trainset_block_count))
                trainset_output_pool5_filename = os.path.join(
                    project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/trainset_feat_pert%d_sp%d_bl%d_pool5.tfd' % (perturb_count, gpu_id, trainset_block_count))
                trainset_output_fc6_filename = os.path.join(
                    project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/trainset_feat_pert%d_sp%d_bl%d_fc6.tfd' % (perturb_count, gpu_id, trainset_block_count))
                trainset_output_fc7_filename = os.path.join(
                    project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/trainset_feat_pert%d_sp%d_bl%d_fc7.tfd' % (perturb_count, gpu_id, trainset_block_count))

                bool_should_run_trainset = True
                if os.path.isfile(trainset_output_pool5_filename) and os.path.isfile(trainset_output_fc6_filename) and os.path.isfile(trainset_output_fc7_filename):
                    bool_should_run_trainset = False
                pass  # end if

                is_training = False if perturb_count == 0 else True

                if bool_should_run_trainset:
                    pool5_feature_matrix, fc6_feature_matrix, fc7_feature_matrix = \
                        extract_vgg_16_features(end_points, sess, image_input,
                                                train_images[trainset_block_index, :], gpu_device_config, real_gpu_device_config, cpu_device_config,
                                                checkpoint_file, perturb_count=perturb_count, is_training=is_training,
                                                )

                    if not os.path.isfile(trainset_output_pool5_filename):
                        distutils.dir_util.mkpath(os.path.dirname(trainset_output_pool5_filename))
                        save_as_tfrecord(trainset_output_pool5_filename, features=pool5_feature_matrix, labels=train_labels[trainset_block_index], unique_image_id=trainset_block_index)

                    if not os.path.isfile(trainset_output_fc6_filename):
                        distutils.dir_util.mkpath(os.path.dirname(trainset_output_fc6_filename))
                        save_as_tfrecord(trainset_output_fc6_filename, features=fc6_feature_matrix, labels=train_labels[trainset_block_index], unique_image_id=trainset_block_index)

                    if not os.path.isfile(trainset_output_fc7_filename):
                        distutils.dir_util.mkpath(os.path.dirname(trainset_output_fc7_filename))
                        save_as_tfrecord(trainset_output_fc7_filename, features=fc7_feature_matrix, labels=train_labels[trainset_block_index], unique_image_id=trainset_block_index)
                pass  # end if bool_should_run_trainset
            pass  # end for perturb_count
            pass
        pass  # end for trainset_block_count

        testset_block_index_list = np.array_split(range(test_images.shape[0]), num_testset_blocks)

        for testset_block_count, testset_block_index in enumerate(testset_block_index_list):
            # testset_output_pool4_filename = os.path.join(project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/testset_feat_sp%d_bl%d_pool4.tfd' % (gpu_id, testset_block_count))
            testset_output_pool5_filename = os.path.join(project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/testset_feat_sp%d_bl%d_pool5.tfd' % (gpu_id, testset_block_count))
            testset_output_fc6_filename = os.path.join(project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/testset_feat_sp%d_bl%d_fc6.tfd' % (gpu_id, testset_block_count))
            testset_output_fc7_filename = os.path.join(project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/testset_feat_sp%d_bl%d_fc7.tfd' % (gpu_id, testset_block_count))

            bool_should_run_testset = True
            if os.path.isfile(testset_output_pool5_filename) and os.path.isfile(testset_output_fc6_filename) and os.path.isfile(testset_output_fc7_filename):
                bool_should_run_testset = False

            if bool_should_run_testset:
                pool5_feature_matrix, fc6_feature_matrix, fc7_feature_matrix = \
                    extract_vgg_16_features(end_points, sess, image_input,
                                            test_images[testset_block_index,:], gpu_device_config, real_gpu_device_config, cpu_device_config, checkpoint_file, is_training=False,
                                            )

                # export to numpy files
                # if not os.path.isfile(testset_output_pool4_filename):
                #     distutils.dir_util.mkpath(os.path.dirname(testset_output_pool4_filename))
                #     save_as_tfrecord(testset_output_pool4_filename, features=pool4_feature_matrix, labels=test_labels)

                if not os.path.isfile(testset_output_pool5_filename):
                    distutils.dir_util.mkpath(os.path.dirname(testset_output_pool5_filename))
                    save_as_tfrecord(testset_output_pool5_filename, features=pool5_feature_matrix, labels=test_labels[testset_block_index], unique_image_id=testset_block_index)

                if not os.path.isfile(testset_output_fc6_filename):
                    distutils.dir_util.mkpath(os.path.dirname(testset_output_fc6_filename))
                    save_as_tfrecord(testset_output_fc6_filename, features=fc6_feature_matrix, labels=test_labels[testset_block_index], unique_image_id=testset_block_index)

                if not os.path.isfile(testset_output_fc7_filename):
                    distutils.dir_util.mkpath(os.path.dirname(testset_output_fc7_filename))
                    save_as_tfrecord(testset_output_fc7_filename, features=fc7_feature_matrix, labels=test_labels[testset_block_index], unique_image_id=testset_block_index)

            pass  # end if bool_should_run_testset
        pass # end for testset_block_count
    pass # end with tf.Graph
    sess.close()

    # build is_training=True
    with tf.Graph().as_default(), tf.device(cpu_device_config):
        # build is_training=False network
        logits, end_points, image_input = build_vgg16_network(gpu_device_config, is_training=True)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, slim.get_model_variables('vgg_16'))
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, ))
        #   Load weights
        init_fn(sess)

        trainset_block_index_list = np.array_split(range(train_images.shape[0]), num_trainset_blocks)

        for trainset_block_count, trainset_block_index in enumerate(trainset_block_index_list):
            for perturb_count in range(1,options.num_perturb):
                # trainset_output_pool4_filename = os.path.join(
                #     project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/trainset_feat_pert%d_sp%d_bl%d_pool4.tfd' % (perturb_count, gpu_id, trainset_block_count))
                trainset_output_pool5_filename = os.path.join(
                    project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/trainset_feat_pert%d_sp%d_bl%d_pool5.tfd' % (perturb_count, gpu_id, trainset_block_count))
                trainset_output_fc6_filename = os.path.join(
                    project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/trainset_feat_pert%d_sp%d_bl%d_fc6.tfd' % (perturb_count, gpu_id, trainset_block_count))
                trainset_output_fc7_filename = os.path.join(
                    project_config.output_dir, 'midlayer_feat_tfrec/cifar10/vgg_16/trainset_feat_pert%d_sp%d_bl%d_fc7.tfd' % (perturb_count, gpu_id, trainset_block_count))

                bool_should_run_trainset = True
                if os.path.isfile(trainset_output_pool5_filename) and os.path.isfile(trainset_output_fc6_filename) and os.path.isfile(trainset_output_fc7_filename):
                    bool_should_run_trainset = False
                pass  # end if

                is_training = False if perturb_count == 0 else True

                if bool_should_run_trainset:
                    pool5_feature_matrix, fc6_feature_matrix, fc7_feature_matrix = \
                        extract_vgg_16_features(end_points, sess, image_input,
                                                train_images[trainset_block_index, :], gpu_device_config, real_gpu_device_config, cpu_device_config,
                                                checkpoint_file, perturb_count=perturb_count, is_training=is_training,
                                                )

                    if not os.path.isfile(trainset_output_pool5_filename):
                        distutils.dir_util.mkpath(os.path.dirname(trainset_output_pool5_filename))
                        save_as_tfrecord(trainset_output_pool5_filename, features=pool5_feature_matrix, labels=train_labels[trainset_block_index], unique_image_id=trainset_block_index)

                    if not os.path.isfile(trainset_output_fc6_filename):
                        distutils.dir_util.mkpath(os.path.dirname(trainset_output_fc6_filename))
                        save_as_tfrecord(trainset_output_fc6_filename, features=fc6_feature_matrix, labels=train_labels[trainset_block_index], unique_image_id=trainset_block_index)

                    if not os.path.isfile(trainset_output_fc7_filename):
                        distutils.dir_util.mkpath(os.path.dirname(trainset_output_fc7_filename))
                        save_as_tfrecord(trainset_output_fc7_filename, features=fc7_feature_matrix, labels=train_labels[trainset_block_index], unique_image_id=trainset_block_index)
                pass  # end if bool_should_run_trainset
            pass  # end for perturb_count
            pass
        pass  # end for trainset_block_count

    pass  # end with tf.Graph
    sess.close()
pass # end def


def save_as_tfrecord(save_filename, features, labels, unique_image_id):
    """
    save features and labels to tfrecord file
    :param save_filename:
    :param features:
    :param labels:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(save_filename)
    for i in range(len(labels)):
        # print('feature_shape=(%d,%d), len_labels=%d' % (features.shape[0], features.shape[1], len(labels)))
        example = tf.train.Example(features=tf.train.Features(feature={  # SequenceExample for seuqnce example
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
            'feature': tf.train.Feature(float_list=tf.train.FloatList(value=features[i,:].tolist() ),),
            'unique_image_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[unique_image_id[i]])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    pass
    writer.close()
pass # end def

if __name__ == '__main__':
    """
    python extract_midlayer_features.py --num_gpus=4 --num_trainset_blocks=250 --num_testset_blocks=25 --num_perturb=2 --debug=True
    """
    parser = OptionParser()
    parser.add_option('--num_gpus', type='int', dest='num_gpus', default=4, help='number of gpu.')
    parser.add_option('--num_trainset_blocks', type='int', dest='num_trainset_blocks', default=128, help='number of data blocks to split the training dataset.')
    parser.add_option('--num_testset_blocks', type='int', dest='num_testset_blocks', default=1, help='number of data blocks to split the testing dataset.')
    parser.add_option('--num_perturb', type='int', dest='num_perturb', default=10, help='number of random perturbation for each image.')
    parser.add_option('--debug', type='string', dest='debug', default=False, help='run debug code.')
    (options, args) = parser.parse_args()
    
    # generate task list
    task_list = []
    for gpu_id in range(options.num_gpus):
        task_list.append([gpu_id, options.num_gpus])
    pass # end for

    par_results = Parallel(n_jobs=options.num_gpus, verbose=50, batch_size=1)(delayed(extract_vgg_16_2016_08_28)(options, par_for_parameters) for par_for_parameters in task_list)