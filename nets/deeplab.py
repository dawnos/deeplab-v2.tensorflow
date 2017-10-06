
import tensorflow as tf
import tensorflow.contrib.slim as slim


def _pad(inputs, padding, name):
  return tf.pad(inputs, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode='CONSTANT', name=name)


def _conv2d_group(inputs, repetitions, filters, pool_stride, dilation_rate, scope_id):
  net = slim.repeat(inputs, repetitions, slim.conv2d,
                    filters, 3, rate=dilation_rate, scope=("conv%d" % scope_id),
                    variables_collections=["pretrained"], outputs_collections=["features"])
  net = _pad(net, 1, name=("pad%d" % scope_id))
  net = slim.max_pool2d(net, 3, stride=pool_stride, scope=('pool%d' % scope_id))
  return net


def _branch(inputs, classes, scope_id, hole):

  with slim.arg_scope([slim.conv2d], outputs_collections=["features"]):
    net = _pad(inputs, hole, 'pad6_%d' % scope_id)
    net = slim.conv2d(net, 1024, 3, rate=hole, padding='VALID', scope=('fc6_%d' % scope_id),
                      variables_collections=["pretrained"])
    net = slim.dropout(net, keep_prob=0.5, scope=('dropout6_%d' % scope_id))
    net = slim.conv2d(net, 1024, 1, scope=('fc7_%d' % scope_id), variables_collections=["pretrained"])
    net = slim.dropout(net, keep_prob=0.5, scope=('dropout7_%d' % scope_id))
    net = slim.conv2d(net, classes, 1, scope=('fc8_EXP_%d' % scope_id), activation_fn=None,
                      variables_collections=["not_pretrained"])
  return net


def deeplab_vgg16(inputs, classes, weight_decay=0.0005, scope='deeplab'):
  with tf.variable_scope(scope):
    with slim.arg_scope([slim.conv2d], weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
      net = _conv2d_group(inputs, 2,  64, pool_stride=2, dilation_rate=1, scope_id=1)
      net = _conv2d_group(   net, 2, 128, pool_stride=2, dilation_rate=1, scope_id=2)
      net = _conv2d_group(   net, 3, 256, pool_stride=2, dilation_rate=1, scope_id=3)
      net = _conv2d_group(   net, 3, 512, pool_stride=1, dilation_rate=1, scope_id=4)
      net = _conv2d_group(   net, 3, 512, pool_stride=1, dilation_rate=2, scope_id=5)

      net_1 = _branch(net, classes, 1,  6)
      net_2 = _branch(net, classes, 2, 12)
      net_3 = _branch(net, classes, 3, 18)
      net_4 = _branch(net, classes, 4, 24)
    with tf.name_scope('fc8'):
      net = net_1 + net_2 + net_3 + net_4

  return net
