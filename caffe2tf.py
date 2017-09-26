
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from caffe.io import blobproto_to_array
from caffe.proto.caffe_pb2 import NetParameter

caffe_file = "/home/tangli/Projects/deeplab/prototxt_and_model/vgg16/init.caffemodel"
tf_file = "/home/tangli/Projects/deeplab/prototxt_and_model/vgg16/init.ckpt"

net_param = NetParameter()
f = open(caffe_file, 'rb')
net_param.ParseFromString(f.read())
f.close()

suffix = ('weights', 'biases')
for i in xrange(len(net_param.layers)):
  layer_name = net_param.layers[i].name
  print layer_name
  for j in xrange(len(net_param.layers[i].blobs)):
    name = layer_name + '/' + suffix[j]
    blob = blobproto_to_array(net_param.layers[i].blobs[j])

    if j == 0:
      # NCHW -> HWCN
      blob = blob.swapaxes(0, 2)
      blob = blob.swapaxes(1, 3)
      blob = blob.swapaxes(2, 3)
    elif j == 1:
      # CHWN -> N
      blob = blob.reshape([blob.shape[3]])
    print str(blob.shape)
    tf.Variable(name=name, expected_shape=blob.shape, initial_value=blob, dtype=tf.float32)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  save_path = saver.save(sess, tf_file, write_meta_graph=False, write_state=False)
print("Model saved in file: %s" % save_path)

print_tensors_in_checkpoint_file(save_path, "", False)