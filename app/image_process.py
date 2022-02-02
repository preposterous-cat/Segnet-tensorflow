import tensorflow as tf
from cv2 import erode
import numpy as np


class ImageProcess:
  __colors_default = [
      (203, 249, 0), #Paru-paru Kanan
      (0, 164, 187), #Paru-paru Kiri
      (204, 204, 204), #Infeksi
      (0, 0, 0) # Background
    ]

  __onehot_default = [
    [1, 0, 0, 0], #Paru-paru Kanan
    [0, 1, 0, 0], #Paru-paru Kiri
    [0, 0, 1, 0], #Infeksi
    [0, 0, 0, 1] #Background
  ] 

  def __init__(self, colors=__colors_default, onehot=__onehot_default):
    self.__colors_default = colors
    self.__onehot_default = onehot
  
  def onehot_mask(self, x):
    mask = x
    mask_onehot = []
    for color in self.__colors_default:
      class_map = tf.reduce_all(tf.equal(mask, color), axis=-1)
      mask_onehot.append(class_map)
    mask_onehot = tf.stack(mask_onehot, axis=-1)
    mask_onehot = tf.cast(mask_onehot, tf.float32)
    mask_onehot = tf.reshape(mask_onehot, shape=(512*512, 4))
    return mask_onehot

  def convertTo(self, tensor, shape, rgb=True):
    if rgb == True:
      palette = tf.constant(self.__colors_default, dtype=tf.uint8)
    else:
      palette = tf.constant(self.__onehot_default, dtype=tf.uint8)
    class_indexes = tf.argmax(tensor, axis=-1)
    class_indexes = tf.reshape(class_indexes, [-1])
    image = tf.gather(palette, class_indexes)
    image = tf.reshape(image, shape)

    return image

  def doErode(self, img):
    kernel = np.ones((2,2), np.uint8)

    img_erosion = erode(img, kernel, iterations=5)
    return img_erosion