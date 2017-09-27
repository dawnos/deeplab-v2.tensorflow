import numpy as np
import tensorflow as tf
from nets.deeplab import deeplab_vgg16
from datasets import pascal
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS

stop_requested = False


def signal_handler(signal, frame):
  print('You pressed Ctrl+C!')
  stop_requested = True


def main(arg):
  classes = 21
  mean = (122.675, 116.669, 104.008)
  max_step = 20000
  save_step = 1000
  momentum = 0.9
  weight_decay = 0.0005
  batch_size = 10

  images, labels = pascal.create_tfrecord_pipeline(
    '/mnt/DataBlock2/VOCdevkit/VOC2012.tfrecord', batch_size = batch_size, crop_size=(321, 321), mean=mean, name="inputs")

  logits = deeplab_vgg16(images, classes=classes, weight_decay=weight_decay)
  resized_prediction = tf.argmax(logits, axis=3)
  resized_prediction = tf.expand_dims(resized_prediction, 3)
  prediction = tf.image.resize_images(resized_prediction, images.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  with tf.name_scope("loss"):
    resized_labels = tf.image.resize_images(labels, logits.get_shape()[1:3], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_labels = tf.cast(resized_labels, tf.int32, name="resized_label")

    logits_row = tf.reshape(logits, [-1, classes])
    resized_labels_row = tf.reshape(resized_labels, [-1])
    valid = tf.squeeze(tf.where(tf.less_equal(resized_labels_row, classes-1)), 1)
    logits_row_valid = tf.gather(logits_row, valid, name="valid_logits")
    resized_labels_row_valid = tf.gather(resized_labels_row, valid, name="valid_labels")
    loss_op = tf.losses.sparse_softmax_cross_entropy(resized_labels_row_valid, logits_row_valid)#  + l2_losses

    resized_prediction_row = tf.reshape(resized_prediction, [-1])
    resized_prediction_row_valid = tf.gather(resized_prediction_row, valid, name="valid_predictions")
    accuracy_op = tf.metrics.accuracy(labels=resized_labels_row_valid, predictions=resized_prediction_row_valid)

  if FLAGS.snapshot != "":
    pretrained_colletion = tf.get_collection("pretrained")
    pretrained_dict = {}
    for var in pretrained_colletion:
      key1 = var.name[0:-2]
      words = key1.split('/')
      key2 = words[-2] + '/' + words[-1]
      pretrained_dict[key2] = var

  else:
    pretrained_dict = []
    pretrained_colletion = []

  with tf.variable_scope("trainer"):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(learning_rate=1e-3, global_step=global_step, decay_steps=max_step,
                                              end_learning_rate=0.0, power=0.9)

    pretrained_weights_collection = tf.get_collection("pretrained", ".*weights.*")
    pretrained_biases_collection = tf.get_collection("pretrained", ".*biases.*")
    not_pretrained_weights_collection = tf.get_collection("not_pretrained", ".*weights.*")
    not_pretrained_biases_collection = tf.get_collection("not_pretrained", ".*biases.*")
    opt_w_op1 = tf.train.MomentumOptimizer(learning_rate * 1, momentum=momentum)
    opt_b_op1 = tf.train.MomentumOptimizer(learning_rate * 2, momentum=momentum)
    opt_w_op2 = tf.train.MomentumOptimizer(learning_rate * 10, momentum=momentum)
    opt_b_op2 = tf.train.MomentumOptimizer(learning_rate * 20, momentum=momentum)
    grads = tf.gradients(loss_op,
                         pretrained_weights_collection +
                         pretrained_biases_collection +
                         not_pretrained_weights_collection +
                         not_pretrained_biases_collection)
    collection_sizes = [len(pretrained_weights_collection),
                        len(pretrained_biases_collection),
                        len(not_pretrained_weights_collection),
                        len(not_pretrained_biases_collection)]
    collection_size_cumsum = [0] + np.cumsum(collection_sizes).tolist()
    each_grads = [grads[collection_size_cumsum[i]:collection_size_cumsum[i+1]] for i in xrange(4)]
    train_w_op1 = opt_w_op1.apply_gradients(zip(each_grads[0], pretrained_weights_collection), global_step=global_step)
    train_b_op1 = opt_b_op1.apply_gradients(zip(each_grads[1], pretrained_biases_collection))
    train_w_op2 = opt_w_op2.apply_gradients(zip(each_grads[2], not_pretrained_weights_collection))
    train_b_op2 = opt_b_op2.apply_gradients(zip(each_grads[3], not_pretrained_biases_collection))
    train_op = tf.group(train_w_op1, train_b_op1, train_w_op2, train_b_op2)

  if FLAGS.display_feature:
    features = tf.get_collection("features")
    for f in features:
      split = tf.split(f, num_or_size_splits=f.get_shape()[3], axis=3)
      tf.summary.image(f.name, split[0], 10)

  if FLAGS.display_fc8:
    logit_chs = tf.split(logits, num_or_size_splits=logits.get_shape()[3], axis=3)
    for logit_ch in logit_chs:
      tf.summary.image('logits', logit_ch)

  # Summary
  tf.summary.scalar("learning_rate", learning_rate)
  tf.summary.scalar("loss", loss_op)
  tf.summary.image("input_images", images)
  tf.summary.image("input_labels", tf.cast(labels * 10, tf.uint8))
  # tf.summary.image("resized_labels", tf.cast(resized_labels * 10, tf.uint8))
  tf.summary.image("predition", tf.cast(prediction * 10, tf.uint8))
  tf.summary.scalar("global_step", global_step)
  merged_summary = tf.summary.merge_all()

  # start session
  sess = tf.Session()

  tf.train.start_queue_runners(sess)

  if FLAGS.debug:
    print 'Entering debug mode...'
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  # Restore
  if FLAGS.snapshot != "":
    print pretrained_dict
    saver = tf.train.Saver(var_list=pretrained_dict)
    print "Restoring pretrained model..."
    saver.restore(sess, FLAGS.snapshot)
    print "Restoring done"

  # Initialization
  print "Initializing variable..."
  # sess.run(tf.global_variables_initializer())
  sess.run(tf.variables_initializer(set(tf.global_variables())-set(pretrained_colletion)))
  print "Initialization done"

  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())

  for step in xrange(0, max_step):
    _, loss, summary = sess.run([train_op, loss_op, merged_summary])
    # _, loss = sess.run([train_op, loss_op])
    print 'loss=%f @ %d/%d' % (loss, step, max_step)

    # Save summary
    summary_writer.add_summary(summary, step)

    # Save checkpoint
    if (step+1) % save_step == 0:
      path = saver.save(sess, FLAGS.save_dir, global_step=global_step)
      print "Checkpoint saved: " + path

  # Save last checkpoint
  path = saver.save(sess, FLAGS.log_dir, global_step=global_step)
  print "Last checkpoint saved: " + path

  # writer.close()
  summary_writer.close()
  sess.close()


if __name__ == "__main__":
  tf.app.flags.DEFINE_string("log_dir", "/tmp/deeplab/log", "Log dir")
  tf.app.flags.DEFINE_string("save_dir", "/tmp/deeplab/model", "Model dir")
  # tf.app.flags.DEFINE_string("snapshot", "", "snapshot dir")
  tf.app.flags.DEFINE_string("snapshot", "/home/tangli/Projects/deeplab/prototxt_and_model/vgg16/init.ckpt",
                             "snapshot dir")
  tf.app.flags.DEFINE_boolean("debug", False, "whether use TFDebug")
  tf.app.flags.DEFINE_boolean("display_feature", False, "whether display_feature")
  tf.app.flags.DEFINE_boolean("display_fc8", False, "whether display_fc8")
  tf.app.run()
