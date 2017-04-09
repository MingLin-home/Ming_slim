""" Convolutional seconder order linear model """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import tensorflow as tf

# slim = tf.contrib.slim
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)

def ConvSLMnet_only_first_order(images, num_classes=10, is_training=False,
                dropout_keep_prob=0.5,
                prediction_fn=slim.softmax,
                weight_decay=0.0001,
                scope='ConvSLMnet'):
  """Creates a convolutional seconder order linear model.

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
  num_cnn_filters = 64
  fc_rank1 = 380
  fc_rank2 = 190

  with tf.variable_scope(scope, 'ConvSLMnet', [images, num_classes]):

    layer1 = slim.conv2d(images, num_cnn_filters, [4, 4], scope='conv1') # 32*32
    layer1 = slim.max_pool2d(layer1, [2, 2], 2, scope='pool1') # 32->16
    layer1 = tf.nn.lrn(layer1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1') # 16*16

    layer2 = slim.conv2d(layer1, num_cnn_filters, [4, 4], scope='conv2') # 16*16
    layer2 = slim.max_pool2d(layer2, [2, 2], 2, scope='pool2') # 16->8
    layer2 = tf.nn.lrn(layer2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2') # 8*8

    layer3 = slim.conv2d(layer2, num_cnn_filters, [4, 4], scope='conv2') # 8*8
    layer3 = slim.max_pool2d(layer3, [2, 2], 2, scope='pool2') # 8->4
    layer3 = tf.nn.lrn(layer3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2') # 4*4

    layer1 = slim.flatten(layer1)
    layer2 = slim.flatten(layer2)
    layer3 = slim.flatten(layer3)


    X = tf.concat([layer2,layer3],axis=1)
    X = slim.dropout(X, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')

    dim_X = num_cnn_filters*(8*8+4*4)

    # Z0 = slim.model_variable('Z0',
    #                          shape=[dim_X, num_classes,W_rank],
    #                          initializer=tf.random_normal_initializer(stddev=0.01/dim_X/num_classes),
    #                          regularizer=slim.l2_regularizer(weight_decay),
    #                          )
    #
    # Z1 = slim.model_variable('Z1',
    #                          shape=[1, 336, num_classes,W_rank],
    #                          initializer=tf.random_normal_initializer(stddev=0.01/dim_X/num_classes),
    #                          regularizer=slim.l2_regularizer(weight_decay),
    #                          )
    # Z2 = slim.model_variable('Z2',
    #                          shape=[1, 1, 24, num_classes, W_rank],
    #                          initializer=tf.random_normal_initializer(stddev=0.01 / dim_X / num_classes),
    #                          regularizer=slim.l2_regularizer(weight_decay),
    #                          )

    W1_matrix = slim.model_variable('W1_matrix',
                                    shape=[dim_X,fc_rank1],
                                    initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(dim_X*fc_rank1)),
                                    regularizer=slim.l2_regularizer(weight_decay),
                                    )
    W2_matrix = slim.model_variable('W2_matrix',
                                    shape=[fc_rank1,fc_rank2],
                                    initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(fc_rank1*fc_rank2)),
                                    regularizer=slim.l2_regularizer(weight_decay),
                                    )

    W3_matrix = slim.model_variable('W3_matrix',
                                    shape=[fc_rank2, num_classes],
                                    initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(fc_rank2*num_classes)),
                                    regularizer=None,
                                    )

    # W_matrix = Z0*Z1
    # W_matrix = tf.reduce_sum(W_matrix,axis=3)
    # W_matrix = tf.reshape(W_matrix,[-1,num_classes])

    Y1 = tf.matmul(tf.matmul(tf.matmul(X,W1_matrix), W2_matrix), W3_matrix)


    bias1 = slim.model_variable('bias1',
                                  shape=[1, 10],
                                  initializer=tf.zeros_initializer(),
                                  regularizer=slim.l2_regularizer(weight_decay), )

    logits = Y1 + bias1
    # logits = Y2 - Y2_diag + bias1

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
pass # end def

ConvSLMnet_only_first_order.default_image_size = 32



def ConvSLMnet(images, num_classes=10, is_training=False,
                dropout_keep_prob=0.5,
                prediction_fn=slim.softmax,
                weight_decay=0.0001,
                scope='ConvSLMnet'):
  """Creates a convolutional seconder order linear model.

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
  num_cnn_filters = 32
  W_rank = 10
  M_rank = 10

  with tf.variable_scope(scope, 'ConvSLMnet', [images, num_classes]):
    net = slim.conv2d(images, 64, [5, 5], scope='conv1')  # 32*32
    end_points['conv1'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')  # 32*32 -> 16*16
    end_points['pool1'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    end_points['conv2'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')  # 16*16 -> 8*8
    end_points['pool2'] = net
    net = slim.flatten(net)
    X = net
    dim_X = 64 * 8 * 8
    
    W_matrix = slim.model_variable('W_matrix',
                                    shape=[dim_X,num_classes],
                                    # initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(dim_X)),
                                    initializer=tf.zeros_initializer(),
                                    regularizer=slim.l2_regularizer(weight_decay),
                                    )
    Y1 = tf.matmul(X,W_matrix)

    U_matrix = slim.model_variable('U_matrix',
                                   shape=[dim_X, M_rank*num_classes],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / math.sqrt(dim_X)),
                                   regularizer=slim.l2_regularizer(weight_decay),
                                   )

    V_matrix = slim.model_variable('V_matrix',
                                   shape=[dim_X, M_rank * num_classes],
                                   # initializer=tf.random_normal_initializer(stddev=0.01 / dim_X / num_classes),
                                   initializer=tf.zeros_initializer(),
                                   regularizer=slim.l2_regularizer(weight_decay),
                                   )

    XU = tf.matmul(X, U_matrix)
    XV = tf.matmul(X, V_matrix)
    Y2 = XU*XV
    Y2 = tf.reshape(Y2,[-1,M_rank,num_classes])
    Y2 = tf.reduce_sum(Y2,axis=1)

    M_diag = tf.reshape(U_matrix*V_matrix,[-1,M_rank,num_classes])
    M_diag = tf.reduce_sum(M_diag,axis=1)

    Y2_diag = tf.matmul((X*X),M_diag)

    bias1 = slim.model_variable('bias1',
                                  shape=[1, 10],
                                  initializer=tf.zeros_initializer(),
                                  regularizer=slim.l2_regularizer(weight_decay), )

    logits = Y1 + Y2 - Y2_diag + bias1
    # logits = Y2 - Y2_diag + bias1

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
pass # end def

ConvSLMnet.default_image_size = 32


def ConvSLMnet_fc34(images, num_classes=10, is_training=False,
               dropout_keep_prob=0.5,
               prediction_fn=slim.softmax,
               weight_decay=0.0001,
               scope='ConvSLMnet'):
  """Creates a convolutional seconder order linear model. Trained SLM after fc1 layer

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
  W_rank = 10
  M_rank = 128
  
  with tf.variable_scope(scope, 'ConvSLMnet', [images, num_classes]):
    net = slim.conv2d(images, 64, [5, 5], scope='conv1')  # 32*32
    end_points['conv1'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')  # 32*32 -> 16*16
    end_points['pool1'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    end_points['conv2'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')  # 16*16 -> 8*8
    end_points['pool2'] = net
    net = slim.flatten(net)
    end_points['Flatten'] = net
    net = slim.fully_connected(net, 384, scope='fc3')
    end_points['fc3'] = net
    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
    #                    scope='dropout3')
    # net = slim.fully_connected(net, 192, scope='fc4')
    # end_points['fc4'] = net
    
    X = net
    dim_X = 384
    
    # W_matrix = slim.model_variable('W_matrix',
    #                                shape=[dim_X, num_classes],
    #                                # initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(dim_X)),
    #                                initializer=tf.zeros_initializer(),
    #                                regularizer=slim.l2_regularizer(weight_decay),
    #                                )
    # Y1 = tf.matmul(X, W_matrix)
    
    U_matrix = slim.model_variable('U_matrix',
                                   shape=[dim_X, M_rank * num_classes],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / math.sqrt(dim_X)),
                                   regularizer=slim.l2_regularizer(weight_decay),
                                   )
    
    V_matrix = slim.model_variable('V_matrix',
                                   shape=[dim_X, M_rank * num_classes],
                                   # initializer=tf.random_normal_initializer(stddev=0.01 / dim_X / num_classes),
                                   initializer=tf.zeros_initializer(),
                                   regularizer=slim.l2_regularizer(weight_decay),
                                   )
    
    XU = tf.matmul(X, U_matrix)
    XV = tf.matmul(X, V_matrix)
    Y2 = XU * XV
    Y2 = tf.reshape(Y2, [-1, M_rank, num_classes])
    Y2 = tf.reduce_sum(Y2, axis=1)
    end_points['Y2'] = Y2
    
    M_diag = tf.reshape(U_matrix * V_matrix, [-1, M_rank, num_classes])
    M_diag = tf.reduce_sum(M_diag, axis=1)
    
    Y2_diag = tf.matmul((X * X), M_diag)
    
    end_points['Y2_diag'] = Y2_diag
    
    bias1 = slim.model_variable('bias1',
                                shape=[1, 10],
                                initializer=tf.zeros_initializer(),
                                regularizer=slim.l2_regularizer(weight_decay), )

    logits = Y2 - Y2_diag + bias1
    # logits = Y1 + Y2 - Y2_diag + bias1
    # logits = Y2 - Y2_diag + bias1
    
    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  
  return logits, end_points


pass  # end def

ConvSLMnet_fc34.default_image_size = 32