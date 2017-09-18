import tensorflow as tf

from nets.deeplab import deeplab_vgg16


def main():
  net = deeplab_vgg16()


if __name__ == "__main__":
  tf.app.run()
