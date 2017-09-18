
from os.path import join
import cv2
import tensorflow as tf
import progressbar
# from Augmentor import Pipeline


# class PascalAugmentor(Pipeline):
#   def __init__(self, images, segmentations):
#     super(PascalAugmentor, self).__init__()
#     self.augmentor_images = images
#     self.class_labels


def convert_to_tfrecords(path):
  sets = ['train', 'val', 'test']
  for s in sets:
    f = open(join(path, 'ImageSets', 'Segmentation', s + '.txt'))
    # img_list = []
    # seg_list = []
    lines = f.readlines()

    bar = progressbar.ProgressBar(len(lines), widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                       progressbar.Percentage(), ' ', progressbar.ETA()])
    bar.start()

    for line in lines:
      line = line.replace('\n', '')
      img_fn = join(path, 'JPEGImages', line + '.jpg')
      seg_fn = join(path, 'SegmentationClass', line + '.png')
      # img_list.append(img_fn)
      # seg_list.append(seg_fn)
      img = cv2.imread(img_fn)
      seg = cv2.imread(seg_fn)
      # cv2.imshow('image', img)
      # cv2.imshow('segmentation', seg)
      # cv2.moveWindow('image', 100, 200)
      # cv2.moveWindow('segmentation', 900, 200)
      # cv2.waitKey()


def create_tfrecord_pipeline():
  pass


if __name__ == "__main__":
  convert_to_tfrecords('/mnt/DataBlock2/VOCdevkit/VOC2012')