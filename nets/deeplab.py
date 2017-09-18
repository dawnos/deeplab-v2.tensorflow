
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import tensorflow.contrib.keras as K
# import tensorflow.contrib.slim.python.slim.nets.vgg


"""
def conv2d_group(input, filters, id):
    net = K.models.Sequential()
    if not (filters is list):
        filters = [filters]
    for i in xrange(0, len(filters)):
        net.add(K.layers.Conv2D(filters[i], 3, activation="ReLU", name=('conv%d_%d' % (id, i+1))))
    net.add(K.layers.MaxPool2D())
    return net(input)
"""


def branch(inputs, classes, scope_id):
  net = slim.conv2d(inputs, 1024, 3, padding=12, rate=12, scope=('fc6_%d' % scope_id))
  net = slim.dropout(net, keep_prob=0.5)
  net = slim.conv2d(net, 1024, 1, scope=('fc7_%d' % scope_id))
  net = slim.dropout(net, keep_prob=0.5)
  net = slim.conv2d(net, classes)
  return net


def deeplab_vgg16(inputs, classes, scope):
  with tf.name_scope(scope):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1')
    net = slim.max_pool2d(net, 2, scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2')
    net = slim.max_pool2d(net, 2, scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, 3, scope='conv3')
    net = slim.max_pool2d(net, 2, scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, 3, scope='conv4')
    net = slim.max_pool2d(net, 2, scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, 3, scope='conv5')
    net = slim.max_pool2d(net, 2, scope='pool5')

    net_1 = branch(net, classes, 1)
    net_2 = branch(net, classes, 2)
    net_3 = branch(net, classes, 3)
    net_4 = branch(net, classes, 4)
    net = net_1 + net_2 + net_3 + net_4

    return net