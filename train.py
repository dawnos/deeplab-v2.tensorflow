import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.deeplab import deeplab_vgg16
from datasets import pascal


FLAGS = tf.app.flags.FLAGS

def main(arg):
  classes = 20
  images, labels = pascal.create_tfrecord_pipeline('/mnt/DataBlock2/VOCdevkit/VOC2012.tfrecord',
                                           batch_size=10, size=(321, 321), mean=(104.008, 116.669, 122.675))
  logits = deeplab_vgg16(images, classes=classes)
  labels = tf.cast(tf.image.resize_images(labels, logits.get_shape()[1:3], tf.image.ResizeMethod.NEAREST_NEIGHBOR), tf.int32)
  valid = tf.where(tf.less_equal(labels, classes))
  logits = tf.gather(logits, valid)
  labels = tf.gather(labels, valid)
  loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

  writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=tf.get_default_graph())

  max_step = 20000
  save_step = 2000
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.polynomial_decay(learning_rate=1e-3, global_step=global_step, decay_steps=max_step, end_learning_rate=0.0, power=0.9)
  train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss, global_step = global_step)

  saver = tf.train.Saver()
  with tf.Session() as sess:

    saver.restore(sess, FLAGS.snapshot)
    sess.run(tf.global_variables_initializer())

    for step in xrange(0, max_step):
      train_op.run()

      if (step+1) % save_step == 0:
        saver.save(sess, FLAGS.log_dir, write_meta_graph=False, write_state=False)

  writer.close()


if __name__ == "__main__":
  tf.app.flags.DEFINE_string("log_dir", "/tmp/deeplab/log", "Log dir")

  tf.app.run()
