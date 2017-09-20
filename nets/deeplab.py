
import tensorflow as tf
import tensorflow.contrib.slim as slim


def pad(inputs, padding):
  return tf.pad(inputs, [[0,0], [padding, padding], [padding, padding], [0,0]], mode='CONSTANT')


def conv2d_group(inputs, repetitions, filters, pool_stride, scope):
  with tf.name_scope(scope):
    net = slim.repeat(inputs, repetitions, slim.conv2d, filters, 3, scope=scope)
    net = pad(net, 1)
    net = slim.max_pool2d(net, 3, stride=pool_stride, scope='pool1')
  return net


def branch(inputs, classes, scope_id, hold):
  net = pad(inputs, hold)
  net = slim.conv2d(net, 1024, 3, rate=hold, padding='VALID', scope=('fc6_%d' % scope_id))
  net = slim.dropout(net, keep_prob=0.5, scope=('dropout6_%d' % scope_id))
  net = slim.conv2d(net, 1024, 1, scope=('fc7_%d' % scope_id))
  net = slim.dropout(net, keep_prob=0.5, scope=('dropout7_%d' % scope_id))
  net = slim.conv2d(net, classes, 1, scope=('fc8_EXP_%d' % scope_id))
  return net


def deeplab_vgg16(inputs, classes, scope='deeplab'):
  with tf.name_scope(scope):
    net = conv2d_group(inputs, 2, 64, 2, 'conv1')
    net = conv2d_group(net, 2, 128, 2, 'conv2')
    net = conv2d_group(net, 3, 256, 2, 'conv3')
    net = conv2d_group(net, 3, 512, 1, 'conv4')
    net = conv2d_group(net, 3, 512, 1, 'conv5')

    net_1 = branch(net, classes, 1, 6)
    net_2 = branch(net, classes, 2, 12)
    net_3 = branch(net, classes, 3, 18)
    net_4 = branch(net, classes, 4, 24)
    with tf.name_scope('fc8'):
      net = net_1 + net_2 + net_3 + net_4

    return net