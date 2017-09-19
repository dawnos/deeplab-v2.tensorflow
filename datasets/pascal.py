
from os.path import join
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
    sets = ['train', 'val', 'test']
    for s in sets:
      f = open(join(path, 'ImageSets', 'Segmentation', s + '.txt'))
      lines = f.readlines()

      bar = progressbar.ProgressBar(len(lines), widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                         progressbar.Percentage(), ' ', progressbar.ETA()])
      bar.start()

      for line in lines:
        line = line.replace('\n', '')
        img_fn = join(path, 'JPEGImages', line + '.jpg')
        seg_fn = join(path, 'SegmentationClass', line + '.png')
        img = cv2.imread(img_fn)
        seg = cv2.imread(seg_fn)
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
      img = cv2.imread(img_fn)
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
        'segmentation_raw': _bytes_feature(seg_str)})
      )
      writer.write(example.SerializeToString())

      bar.update(i+1)

    writer.close()


def create_tfrecord_pipeline():
  pass


if __name__ == "__main__":
  convert_to_tfrecords('/mnt/DataBlock2/VOCdevkit/VOC2012', listfile='/mnt/DataBlock2/deeplab_list/train_aug.txt')