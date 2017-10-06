import numpy as np
from time import sleep
import Image
import cv2
import tensorflow as tf
from nets.deeplab import deeplab_vgg16
from datasets import pascal
from tensorflow.python import debug as tf_debug
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

FLAGS = tf.app.flags.FLAGS


def decode_label(label):
  return tf.cast(tf.clip_by_value(tf.cast(label, tf.float32) * 10, clip_value_min=0, clip_value_max=255), tf.uint8)


def main(arg):
  classes = 21
  mean = (122.675, 116.669, 104.008)
  # max_step = 20000
  # test_step = 0
  test_interval = 200000
  save_step = 1000
  momentum = 0.9
  weight_decay = 0.0005

  images, labels, image_fns, original_shapes = pascal.create_pipeline(
    FLAGS.input, root_dir=FLAGS.root_dir,
    batch_size=FLAGS.batch_size, crop_size=(FLAGS.crop_size, FLAGS.crop_size), mean=mean,
    shuffle=FLAGS.shuffle, name="inputs")
  logits = deeplab_vgg16(images, classes=classes, weight_decay=weight_decay)
  resized_prediction = tf.expand_dims(tf.argmax(logits, axis=3), 3)
  resized_prediction = tf.cast(resized_prediction, tf.int32, name="resized_prediction")
  prediction = tf.image.resize_images(resized_prediction, images.get_shape()[1:3],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  with tf.name_scope("loss"):
    with tf.name_scope('resized_label'):
      resized_labels = tf.image.resize_images(labels, logits.get_shape()[1:3], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_labels = tf.cast(resized_labels, tf.int32)

    logits_row = tf.reshape(logits, [-1, classes])
    resized_labels_row = tf.reshape(resized_labels, [-1])
    valid = tf.squeeze(tf.where(tf.less_equal(resized_labels_row, classes-1)), 1)
    logits_row_valid = tf.gather(logits_row, valid, name="valid_logits")
    resized_labels_row_valid = tf.gather(resized_labels_row, valid, name="valid_labels")
    loss_op = tf.losses.sparse_softmax_cross_entropy(resized_labels_row_valid, logits_row_valid)

    resized_prediction_row = tf.reshape(resized_prediction, [-1])
    resized_prediction_row_valid = tf.gather(resized_prediction_row, valid, name="valid_predictions")
  accuracy_op = tf.reduce_mean(tf.cast(tf.equal(resized_prediction_row_valid, resized_labels_row_valid), tf.float32), name='accuracy')

  with tf.variable_scope("trainer"):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(learning_rate=1e-3, global_step=global_step, decay_steps=FLAGS.max_step,
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
  tf.summary.image("input_labels", decode_label(labels))
  tf.summary.image("prediction", decode_label(prediction))
  tf.summary.scalar("global_step", global_step)
  merged_summary = tf.summary.merge_all()

  # start session
  sess = tf.Session()

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord=coord)

  if FLAGS.debug:
    print 'Entering debug mode...'
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  # Restore
  if FLAGS.snapshot != "":
    ckpt_reader = NewCheckpointReader(FLAGS.snapshot)
    var_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    ckpt_var_names = var_to_shape_map.keys()
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    restore_dict = {}
    for tv in trainable_vars:
      for vn in ckpt_var_names:
        if tv.name.find(vn) != -1:
          restore_dict[vn] = tv

    if len(restore_dict) != 0:
      saver = tf.train.Saver(var_list=restore_dict)
      print "Restoring pretrained model..."
      print sorted(restore_dict.keys())
      saver.restore(sess, FLAGS.snapshot)
      print "Restoring done"
  else:
    restore_dict = {}

  # Initialization
  print "Initializing variable..."
  uninitialized_var = set(tf.global_variables())-set(restore_dict.values())
  print uninitialized_var
  sess.run(tf.variables_initializer(uninitialized_var))
  print "Initialization done"

  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.log_path, graph=tf.get_default_graph())

  step = 0
  while True:

    # Test
    if (step % test_interval) == 0:
      accuracy_total = 0.0
      accuracy_count = 0
      for ti in xrange(0, FLAGS.test_step):
        accuracy, pred, _image_fns, _logits, _merged_summary, _images = \
          sess.run([accuracy_op, tf.cast(resized_prediction, tf.uint8), image_fns, logits, merged_summary, tf.cast(images, tf.uint8)])
        accuracy_total += accuracy
        accuracy_count += 1

        if FLAGS.test_result_dir != "":
          assert(pred.shape[0] == _image_fns.shape[0])
          for i in xrange(pred.shape[0]):
            fn = _image_fns[i]
            img = Image.fromarray(np.array(Image.open(fn)))
            ss = img.size
            ww = ss[0]
            hh = ss[1]

            width_diff = FLAGS.crop_size - ww
            # offset_crop_width = max(-width_diff // 2, 0)
            offset_pad_width = max(width_diff // 2, 0)

            height_diff = FLAGS.crop_size - hh
            # offset_crop_height = max(-height_diff // 2, 0)
            offset_pad_height = max(height_diff // 2, 0)

            pp = Image.fromarray(pred[i, :, :, 0])
            pp = pp.resize((FLAGS.crop_size, FLAGS.crop_size), resample=Image.NEAREST)
            pp = pp.crop(box=(offset_pad_width, offset_pad_height, offset_pad_width+ww, offset_pad_height+hh))

            # ppp = Image.fromarray(_images[i, :, :, :])
            # ppp = ppp.crop()

            # pp = pp[i, offset_pad_height:(offset_pad_height+hh), offset_pad_width:(offset_pad_width+ww), 0]

            fn = fn.split("/")
            fn = fn[-1]
            fn = fn.replace('jpg', 'png')
            pp.save(FLAGS.test_result_dir + '/' + fn)
            # ppp.save(FLAGS.test_result_dir + '/' + fn + '.jpg')

            # Image.fromarray(np.array(pp)).show(title='pp')

            # img.show(title='img')
            # sleep(10)
            # Image.

      if accuracy_count > 0:
        accuracy = accuracy_total / accuracy_count
      else:
        accuracy = 0.0
      print 'Accuracy:' + str(accuracy)

    if step >= FLAGS.max_step:
      break

    _, loss, summary = \
      sess.run([train_op, loss_op, merged_summary])
    # _, loss = sess.run([train_op, loss_op])
    print 'loss=%f @ %d/%d' % (loss, step, FLAGS.max_step)

    # Save summary
    summary_writer.add_summary(summary, step)

    # Save checkpoint
    if (step+1) % save_step == 0:
      path = saver.save(sess, FLAGS.save_path, global_step=global_step)
      print "Checkpoint saved: " + path

    step += 1

  # Save last checkpoint
  if step > 0:
    path = saver.save(sess, FLAGS.save_path, global_step=global_step)
    print "Last checkpoint saved: " + path

  summary_writer.close()
  coord.request_stop()
  coord.join(threads)
  sess.close()


if __name__ == "__main__":
  tf.app.flags.DEFINE_string("input",
                             # "/mnt/DataBlock2/VOCdevkit/VOC2012.tfrecord",
                             # "/mnt/DataBlock2/VOCdevkit/VOC2012.val.tfrecord",
                             "/mnt/DataBlock2/deeplab_list/val.txt",
                             "Input")
  tf.app.flags.DEFINE_string("root_dir",
                             "/mnt/DataBlock2/VOCdevkit/VOC2012",
                             # "",
                             "root_dir")
  tf.app.flags.DEFINE_string("log_path", "/tmp/deeplab/log", "Log path")
  tf.app.flags.DEFINE_string("save_path", "/tmp/deeplab/model", "Model path")
  tf.app.flags.DEFINE_string("snapshot",
                             # "/home/tangli/Projects/deeplab/prototxt_and_model/vgg16/init.ckpt",
                             "/tmp/deeplab/model-20000",
                             # "",
                             "snapshot dir")
  tf.app.flags.DEFINE_boolean("debug", False, "whether use TFDebug")
  tf.app.flags.DEFINE_boolean("display_feature", False, "whether display_feature")
  tf.app.flags.DEFINE_boolean("display_fc8", False, "whether display_fc8")
  tf.app.flags.DEFINE_boolean("shuffle", False, "whether shuffle the input")
  tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
  tf.app.flags.DEFINE_integer("crop_size", 513, "crop size")
  tf.app.flags.DEFINE_integer("max_step", 0, "max step")
  tf.app.flags.DEFINE_integer("test_step", 145, "test step")
  tf.app.flags.DEFINE_string("test_result_dir",
                             "/mnt/DataBlock2/VOCdevkit/results/VOC2012/Segmentation/comp5_val_cls/",
                             # "",
                             "test_result_dir")
  tf.app.run()
