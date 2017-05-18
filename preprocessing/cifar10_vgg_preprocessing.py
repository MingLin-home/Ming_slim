# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images in CIFAR-10.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_PADDING = 4
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

slim = tf.contrib.slim


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def preprocess_for_train(image,
                         output_height,
                         output_width,
                        R_mean=0, G_mean=0, B_mean=0,padding=_PADDING,):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    padding: The amound of padding before and after each dimension of the image.

  Returns:
    A preprocessed image.
  """
  # tf.summary.image('image', tf.expand_dims(image, 0))

  # Transform the image to floats.
  image = tf.to_float(image)
  if padding > 0:
    image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])

  # image = tf.image.resize_images(image,(output_height,output_width))

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(image,
                                   [32, 32, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # distorted_image = tf.image.random_brightness(distorted_image,
  #                                              max_delta=63)
  # distorted_image = tf.image.random_contrast(distorted_image,
  #                                            lower=0.2, upper=1.8)
  # Subtract off the mean and divide by the variance of the pixels.
  
  distorted_image = tf.image.resize_images(distorted_image,[output_height, output_width])
  # distorted_image = tf.image.per_image_standardization(distorted_image)

  distorted_image = _mean_image_subtraction(distorted_image, [R_mean,G_mean,B_mean])
  
  return distorted_image


def preprocess_for_eval(image, output_height, output_width,R_mean,G_mean,B_mean):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
  # tf.summary.image('image', tf.expand_dims(image, 0))
  # Transform the image to floats.
  image = tf.to_float(image)
  # image = tf.image.resize_images(image, (output_height, output_width))

  # Resize and crop if needed.
  distorted_image = tf.image.resize_images(image, [output_height, output_width])
  # resized_image = tf.image.resize_image_with_crop_or_pad(image,
  #                                                        output_width,
  #                                                        output_height)
  # tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))

  # Subtract off the mean and divide by the variance of the pixels.
  # distorted_image = tf.image.per_image_standardization(distorted_image)
  distorted_image = _mean_image_subtraction(distorted_image, [R_mean,G_mean,B_mean])
  return distorted_image


def preprocess_image(image, output_height, output_width, is_training=False, R_mean=_R_MEAN, G_mean=_G_MEAN,B_mean=_B_MEAN):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, output_height, output_width, R_mean, G_mean, B_mean)
  else:
    return preprocess_for_eval(image, output_height, output_width, R_mean, G_mean, B_mean)
