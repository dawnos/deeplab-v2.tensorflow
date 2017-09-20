
import tensorflow as tf
from tensorflow.core.framework import variable_pb2, tensor_pb2, graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2, tensor_bundle_pb2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import caffe
from caffe.proto import caffe_pb2

# proto_file = "/tmp/tensorflow/dcgan/log/model.ckpt-15000.meta"
# proto_file = "/tmp/tensorflow/dcgan/log/model.ckpt-13500.data-00000-of-00001"
# proto_file = "/tmp/tensorflow/dcgan/log/model.ckpt-13500"
proto_file = "/home/tangli/Projects/deeplab/prototxt_and_model/vgg16/init.caffemodel"

# print_tensors_in_checkpoint_file(proto_file, "", True)

# m_def = variable_pb2.VariableDef()
# m_def = tensor_pb2.TensorProto()
# m_def = tensor_bundle_pb2.BundleEntryProto()
# m_def = tensor_bundle_pb2.BundleHeaderProto()
# m_def = graph_pb2.GraphDef()
# m_def = meta_graph_pb2.MetaGraphDef()
# m_def = caffe_pb2.NetParameter()

# f = tf.gfile.GFile(proto_file, 'rb').read()
# with open(proto_file, "rb") as f:
#   print m_def.ParseFromString(f.read())

# for field, value in m_def.ListFields():
#   print "name:" + field.name
#   print "type:" + str(field.type)
#   print "label:" + str(field.label)

  # if field.type == 11:
  #   print dir(value)
  #   layer = caffe_pb2.LayerParameter()
  #   layer.ParseFromString(value)

net_param = caffe_pb2.NetParameter()
f = open(proto_file, 'rb')
net_param.ParseFromString(f.read())
f.close()

suffix = ('w', 'b')
for i in xrange(len(net_param.layers)):
  layer_name = net_param.layers[i].name
  print layer_name
  for j in xrange(len(net_param.layers[i].blobs)):
    name = layer_name + '/' + suffix[j]
    # shape = (net_param.layers[i].blobs[j].num,
    #          net_param.layers[i].blobs[j].channels,
    #          net_param.layers[i].blobs[j].height,
    #          net_param.layers[i].blobs[j].width)
    # data_len = len(net_param.layers[i].blobs[j].data)
    blob = caffe.io.blobproto_to_array(net_param.layers[i].blobs[j])
    print str(blob.shape)
    # print type(blob)
    tf.Variable(name=name, expected_shape=blob.shape, initial_value=blob, dtype=tf.float32)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  save_path = saver.save(sess, "/tmp/init.ckpt")
print("Model saved in file: %s" % save_path)

print_tensors_in_checkpoint_file(save_path, "", True)