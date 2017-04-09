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

def lowrank_matrix_net(images, num_classes=10, is_training=False,
                dropout_keep_prob=0.5,
                prediction_fn=slim.softmax,
                weight_decay=0.0001,
                scope='lowrank_matrix_net'):
  """ y = image*W, 

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

  with tf.variable_scope(scope, 'lowrank_matrix_net', [images, num_classes]):
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
    net = slim.flatten(net)
    X = net
    dim_X = 64*8*8

    W = slim.model_variable('W_matrix',
                            shape=[dim_X,num_classes],
                            initializer=tf.zeros_initializer(),
                            regularizer=None,
                            )
    Y1 = tf.matmul(X,W)

    bias1 = slim.model_variable('bias1',
                                  shape=[1, 10],
                                  initializer=tf.zeros_initializer(),
                                  regularizer=None, )

    logits = Y1 + bias1
    # logits = Y2 - Y2_diag + bias1

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
pass # end def

lowrank_matrix_net.default_image_size = 32

def tensor_fold_net_4p6(images, num_classes=10, is_training=False,
                dropout_keep_prob=0.5,
                prediction_fn=slim.softmax,
                weight_decay=0.0001,
                scope='lowrank_matrix_net'):
  """ y = image*U*V, where each column of U is a flattened tensor. U is flattened from 4/4/4/4/4/3 (3072 dimensions) 

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


  latent_feat_dim = 256
  U_column_rank = 64

  with tf.variable_scope(scope, 'lowrank_matrix_net', [images, num_classes]):
    net = slim.conv2d(images, 64, [5, 5], scope='conv1') # 32*32
    end_points['conv1'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1') # 32*32 -> 16*16
    end_points['pool1'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    end_points['conv2'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2') # 16*16 -> 8*8
    end_points['pool2'] = net
    net = slim.flatten(net)
    X = net
    dim_X = 64*8*8

    init_Gaussian_std = 0.5

    Z1 = slim.model_variable('Z1_matrix',
                            shape=[4,1,1,1,1,1, latent_feat_dim,  U_column_rank,],
                            initializer=tf.random_normal_initializer(stddev=init_Gaussian_std),
                            regularizer=slim.l2_regularizer(weight_decay),
                            )
    # Z1 = tf.reshape(Z1,[4,-1])
    # Z1 = slim.unit_norm(Z1,0)
    # Z1 = tf.reshape(Z1,[4,1,1,1,1,1, latent_feat_dim,  U_column_rank,])
    
    Z2 = slim.model_variable('Z2_matrix',
                             shape=[1, 4, 1, 1, 1, 1, latent_feat_dim,  U_column_rank, ],
                             initializer=tf.random_normal_initializer(stddev=init_Gaussian_std),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    # Z2 = tf.reshape(Z2,[1,4,-1])
    # Z2 = slim.unit_norm(Z2, 1)
    # Z2 = tf.reshape(Z2,[1, 4, 1, 1, 1, 1, latent_feat_dim,  U_column_rank, ])
    
    Z3 = slim.model_variable('Z3_matrix',
                             shape=[1, 1, 4, 1, 1, 1, latent_feat_dim,  U_column_rank, ],
                             initializer=tf.random_normal_initializer(stddev=init_Gaussian_std),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    # Z3 = tf.reshape(Z3, [1, 4, -1])
    # Z3 = slim.unit_norm(Z3, 1)
    # Z3 = tf.reshape(Z3, [1, 1, 4, 1, 1, 1, latent_feat_dim, U_column_rank, ])
    
    Z4 = slim.model_variable('Z4_matrix',
                             shape=[1, 1, 1, 4, 1, 1, latent_feat_dim,  U_column_rank, ],
                             initializer=tf.random_normal_initializer(stddev=init_Gaussian_std),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    # Z4 = tf.reshape(Z4, [1, 4, -1])
    # Z4 = slim.unit_norm(Z4, 1)
    # Z4 = tf.reshape(Z4, [1, 1, 1, 4, 1, 1, latent_feat_dim, U_column_rank, ])
    
    
    Z5 = slim.model_variable('Z5_matrix',
                             shape=[1, 1, 1, 1, 4, 1, latent_feat_dim,  U_column_rank, ],
                             initializer=tf.random_normal_initializer(stddev=init_Gaussian_std),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    # Z5 = tf.reshape(Z5, [1, 4, -1])
    # Z5 = slim.unit_norm(Z5, 1)
    # Z5 = tf.reshape(Z5, [1, 1, 1, 1, 4, 1, latent_feat_dim, U_column_rank, ])
    
    Z6 = slim.model_variable('Z6_matrix',
                             shape=[1, 1, 1, 1, 1, 4, latent_feat_dim,  U_column_rank, ],
                             initializer=tf.random_normal_initializer(stddev=init_Gaussian_std),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    # Z6 = tf.reshape(Z6, [1, 4, -1])
    # Z6 = slim.unit_norm(Z6, 1)
    # Z6 = tf.reshape(Z6, [1, 1, 1, 1, 1, 4, latent_feat_dim, U_column_rank, ])

    U = Z1*Z2*Z3*Z4*Z5*Z6
    U = tf.reshape(U, [dim_X, latent_feat_dim, U_column_rank])
    U = tf.reduce_sum(U,axis=2)
    
    # U = slim.batch_norm(U)
    

    V = slim.model_variable('V_matrix',
                            shape=[latent_feat_dim, num_classes],
                            # initializer=tf.random_normal_initializer(stddev=0.01),
                            initializer=tf.zeros_initializer(),
                            regularizer=slim.l2_regularizer(weight_decay),
                            )

    Y1 = tf.matmul(X,U)
    # Y1 = slim.batch_norm(Y1)
    Y2 = tf.matmul(Y1, V)

    bias1 = slim.model_variable('bias1',
                                  shape=[1, 10],
                                  initializer=tf.zeros_initializer(),
                                  regularizer=slim.l2_regularizer(weight_decay), )

    logits = Y2 + bias1
    # logits = Y2 - Y2_diag + bias1

    # UTU_minus_I = tf.matmul(U,U,transpose_a=True) - tf.eye(latent_feat_dim)
    #
    # U_orthogonal_regularizer_loss = weight_decay* tf.norm(UTU_minus_I)**2
    # tf.losses.add_loss(U_orthogonal_regularizer_loss)

    # tf.summary.scalar('losses/UTU_orthogonal_loss',U_orthogonal_regularizer_loss)

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
pass # end def

tensor_fold_net_4p6.default_image_size = 32


def tensor_fold_net_64times64(images, num_classes=10, is_training=False,
                        dropout_keep_prob=0.5,
                        prediction_fn=slim.softmax,
                        weight_decay=0.0001,
                        scope='lowrank_matrix_net'):
  """ y = image*U*V, where each column of U is a flattened tensor. U is flattened from 4/4/4/4/4/3 (3072 dimensions)

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
  
  latent_feat_dim = 128
  U_column_rank = 256
  
  with tf.variable_scope(scope, 'lowrank_matrix_net', [images, num_classes]):
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
    
    init_Gaussian_std = 1.0/8.0
    
    Z1 = slim.model_variable('Z1_matrix',
                             shape=[64, 1, latent_feat_dim, U_column_rank, ],
                             initializer=tf.random_normal_initializer(stddev=init_Gaussian_std),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    
    Z2 = slim.model_variable('Z2_matrix',
                             shape=[1, 64, latent_feat_dim, U_column_rank, ],
                             initializer=tf.random_normal_initializer(stddev=init_Gaussian_std),
                             regularizer=slim.l2_regularizer(weight_decay),
                             )
    
    U = Z1 * Z2
    U = tf.reshape(U, [dim_X, latent_feat_dim, U_column_rank])
    U = tf.reduce_sum(U, axis=2)
    
    # U = slim.batch_norm(U)
    
    
    V = slim.model_variable('V_matrix',
                            shape=[latent_feat_dim, num_classes],
                            # initializer=tf.random_normal_initializer(stddev=0.01),
                            initializer=tf.zeros_initializer(),
                            regularizer=slim.l2_regularizer(weight_decay),
                            )
    
    Y1 = tf.matmul(X, U)
    # Y1 = slim.batch_norm(Y1)
    Y2 = tf.matmul(Y1, V)
    
    bias1 = slim.model_variable('bias1',
                                shape=[1, 10],
                                initializer=tf.zeros_initializer(),
                                regularizer=slim.l2_regularizer(weight_decay), )
    
    logits = Y2 + bias1
    # logits = Y2 - Y2_diag + bias1
    
    # UTU_minus_I = tf.matmul(U,U,transpose_a=True) - tf.eye(latent_feat_dim)
    #
    # U_orthogonal_regularizer_loss = weight_decay* tf.norm(UTU_minus_I)**2
    # tf.losses.add_loss(U_orthogonal_regularizer_loss)
    
    # tf.summary.scalar('losses/UTU_orthogonal_loss',U_orthogonal_regularizer_loss)
    
    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  
  return logits, end_points


pass  # end def

tensor_fold_net_64times64.default_image_size = 32

def ming_arg_scope(weight_decay=1e-12):
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
        weights_initializer=trunc_normal(0.04),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu) as sc:
      return sc