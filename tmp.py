
import tensorflow as tf
from tensorflow.core.framework import variable_pb2, tensor_pb2, graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2, tensor_bundle_pb2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# import caffe
# from caffe.proto import caffe_pb2

# proto_file = "/tmp/tensorflow/dcgan/log/model.ckpt-15000.meta"
proto_file = "/tmp/tensorflow/dcgan/log/model.ckpt-13500.data-00000-of-00001"
# proto_file = "/home/tangli/Projects/deeplab/prototxt_and_model/vgg16/init.caffemodel"

# print_tensors_in_checkpoint_file(proto_file, "", True)

# m_def = variable_pb2.VariableDef()
# m_def = tensor_pb2.TensorProto()
# m_def = tensor_bundle_pb2.BundleEntryProto()
m_def = tensor_bundle_pb2.BundleHeaderProto()
# m_def = graph_pb2.GraphDef()
# m_def = meta_graph_pb2.MetaGraphDef()
# m_def = caffe_pb2.NetParameter()

# f = tf.gfile.GFile(proto_file, 'rb').read()
with open(proto_file, "rb") as f:
  m_def.ParseFromString(f.read())

# for field, value in m_def.ListFields():
#   print "name:" + field.name
#   print "type:" + str(field.type)
#   print "label:" + str(field.label)

  # if field.type == 11:
  #   print dir(value)
  #   layer = caffe_pb2.LayerParameter()
  #   layer.ParseFromString(value)
