"""A simple single layer bilinear net """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# slim = tf.contrib.slim
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def BilinearNet(images, num_classes=10, is_training=False,
                dropout_keep_prob=0.5,
                rank_k1=8,
                rank_k2=256,
                rank_k3=10,
                prediction_fn=slim.softmax,
                weight_decay=0.0001,
                scope='BilinearNet'):
  """Creates a BilinearNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = cifarnet.cifarnet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """


  end_points = {}

  with tf.variable_scope(scope, 'BilinearNet', [images, num_classes]):
    net = slim.conv2d(images, 64, [5, 5], scope='conv1')
    end_points['conv1'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    end_points['pool1'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    end_points['conv2'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    end_points['pool2'] = net

    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
    #                    scope='dropout3')

    # U_matrix = slim.model_variable('U_matrix',
    #                           shape=[rank_k2,rank_k3],
    #                           initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                           regularizer=slim.l2_regularizer(weight_decay),
    #                           )
    # V_matrix = slim.model_variable('V_matrix',
    #                           shape=[rank_k3,num_classes],
    #                           initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                           regularizer=slim.l2_regularizer(weight_decay),
    #                           )

    UV_matrix = slim.model_variable('UV_matrix',
                                    shape=[rank_k2,num_classes],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    regularizer=slim.l2_regularizer(weight_decay),
                                    )



    # transform_T1_list = []
    # for k2 in range(rank_k2): # construct one column of transform_T_matrix
    #   transform_T1_tosum_list = []
    #   for k1 in range(rank_k1):
    #     z1 = slim.model_variable('z1_%d_%d' % (k1,k2),
    #                                  shape=[64,1,1],
    #                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                                  regularizer=slim.l2_regularizer(weight_decay),
    #                                  )
    #
    #     z3 = slim.model_variable('z3_%d_%d' % (k1, k2),
    #                              shape=[1, 16, 1],
    #                              initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                              regularizer=slim.l2_regularizer(weight_decay),
    #                              )
    #     z4 = slim.model_variable('z4_%d_%d' % (k1, k2),
    #                              shape=[1, 1, 16],
    #                              initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                              regularizer=slim.l2_regularizer(weight_decay),
    #                              )
    #
    #     transform_T1_tosum_list.append(z1*z3*z4)
    #     pass # end if
    #   pass # end for k1
    #   transform_T1 = tf.add_n(transform_T1_tosum_list)
    #   transform_T1 = tf.reshape(transform_T1,[-1,1])
    #   transform_T1_list.append(transform_T1)
    # pass # end for k2

    # Z1 = slim.model_variable('Z1',
    #                          shape=[64,1,1,rank_k2,rank_k1],
    #                          initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                          regularizer=slim.l2_regularizer(weight_decay),
    #                          )
    #
    # Z2 = slim.model_variable('Z2',
    #                          shape=[1, 8, 1, rank_k2,rank_k1],
    #                          initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                          regularizer=slim.l2_regularizer(weight_decay),
    #                          )
    # Z3 = slim.model_variable('Z3',
    #                          shape=[1, 1, 8, rank_k2, rank_k1],
    #                          initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                          regularizer=slim.l2_regularizer(weight_decay),
    #                          )
    #
    # transform_T_matrix = Z1*Z2*Z3
    # transform_T_matrix = tf.reduce_sum(transform_T_matrix,axis=4)
    # transform_T_matrix = tf.reshape(transform_T_matrix,[64*8*8,rank_k2])

    Z1 = slim.model_variable('Z1',
                             shape=[8,1,1,1,rank_k2,rank_k1],
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )

    Z2 = slim.model_variable('Z2',
                             shape=[1, 8, 1, 1, rank_k2,rank_k1],
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    Z3 = slim.model_variable('Z3',
                             shape=[1, 1, 8, 1, rank_k2, rank_k1],
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    Z4 = slim.model_variable('Z3',
                             shape=[1, 1, 1, 8, rank_k2, rank_k1],
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )

    transform_T_matrix = Z1*Z2*Z3*Z4
    transform_T_matrix = tf.reduce_sum(transform_T_matrix,axis=5)
    transform_T_matrix = tf.reshape(transform_T_matrix,[64*8*8,rank_k2])



    # end_points['transform_T_matrix'] = transform_T_matrix


    net = tf.reshape(net,[-1,64*8*8])
    net = tf.matmul(net,transform_T_matrix)
    # end_points['after_transform_T_features'] = net

    UV_bias = slim.model_variable('UV_bias',
                                 shape=[1, 10],
                                 initializer=tf.zeros_initializer(),
                                 regularizer=slim.l2_regularizer(weight_decay), )
    # net = tf.matmul(net,U_matrix)
    # net = tf.matmul(net, V_matrix)
    net = tf.matmul(net, UV_matrix)
    net += UV_bias

    logits = net

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
pass # end def

BilinearNet.default_image_size = 32



def BilinearNet_arg_scope(weight_decay=0.0001):
  """Defines the default cifarnet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d],
      weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
      activation_fn=tf.nn.relu):
    with slim.arg_scope(
        [slim.fully_connected],
        biases_initializer=tf.constant_initializer(0.1),
        weights_initializer=trunc_normal(0.04),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu) as sc:
      return sc