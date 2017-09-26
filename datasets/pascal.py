
import cv2
import tensorflow as tf
import progressbar
import Image
import numpy as np


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecords(path, output_filename=None, listfile=None, preprocess=False):

  if output_filename is None:
    output_filename = path + ".tfrecord"
  writer = tf.python_io.TFRecordWriter(output_filename)

  if listfile is None:
    raise Exception('Not implemented')

  else:
    f = open(listfile, 'r')
    lines = f.readlines()

    bar = progressbar.ProgressBar(len(lines), widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                       progressbar.Percentage(), ' ', progressbar.ETA()])
    bar.start()
    for i in xrange(0, len(lines)):
      line = lines[i]
      line = line.replace('\n', '')
      line = line.split(' ')
      img_fn = path + line[0]
      seg_fn = path + line[1]
      # img = cv2.imread(img_fn)
      # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = np.array(Image.open(img_fn))
      seg = np.array(Image.open(seg_fn))

      if img is None:
        raise Exception("Cannot read image " + img_fn)

      if seg is None:
        raise Exception("Cannot read segmentation " + seg_fn)

      assert (img.shape[0:2] == seg.shape[0:2])

      """
      cv2.imshow('image', img)
      cv2.imshow('segmentation', seg*10)
      cv2.moveWindow('image', 100, 200)
      cv2.moveWindow('segmentation', 900, 200)
      cv2.waitKey()
      """

      if preprocess:
        raise Exception("Not implemented")

      img_str = img.tostring()
      seg_str = seg.tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(img.shape[0]),
        'width': _int64_feature(img.shape[1]),
        'depth': _int64_feature(img.shape[2]),
        'image_raw': _bytes_feature(img_str),
        'segmentation_raw': _bytes_feature(seg_str),
      }))
      writer.write(example.SerializeToString())

      bar.update(i+1)

    writer.close()


def create_tfrecord_pipeline(filename, batch_size=64, crop_size=[64, 64], mean=None, name="pipeline"):
  with tf.variable_scope(name):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'segmentation_raw': tf.FixedLenFeature([], tf.string),
      })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    # depth = tf.cast(features['depth'], tf.int32)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, tf.stack([height, width, 3]))
    image = tf.cast(image, tf.float32)

    label = tf.decode_raw(features['segmentation_raw'], tf.uint8)
    label = tf.reshape(label, tf.stack([height, width, 1]))
    label = tf.cast(label, tf.float32)

    # Concat to make sure same operation are appiled to image and label
    concat = tf.concat([image, label-255], 2)
    image_shape = tf.shape(image)
    concat = tf.image.pad_to_bounding_box(concat, 0, 0,
                                          tf.maximum(crop_size[0], image_shape[0]),
                                          tf.maximum(crop_size[1], image_shape[1]))

    # Random rescale
    # concat = tf.image.resize_images(concat, tf.random_uniform(2, 0.5, 1.5) * crop_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Random crop
    concat = tf.random_crop(concat, [crop_size[0], crop_size[1], 4])

    # Random mirror
    concat = tf.image.flip_left_right(concat)

    # Split image and label
    image = concat[:, :, :3]
    label = concat[:, :, 3:] + 255
    label = tf.cast(label, dtype=tf.uint8)

    # Mean substraction
    if not (mean is None):
      if mean is list or mean is tuple:
        mean = np.array(mean)
      image = tf.subtract(image, mean)

    [images, labels] = tf.train.shuffle_batch(
      [image, label], batch_size=batch_size, num_threads=4,
      capacity=1000 + 3 * batch_size, min_after_dequeue=1000)

    return images, labels


if __name__ == "__main__":
  convert_to_tfrecords('/mnt/DataBlock2/VOCdevkit/VOC2012', listfile='/mnt/DataBlock2/deeplab_list/train_aug.txt')