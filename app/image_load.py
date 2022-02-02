import tensorflow as tf

class ImageLoad:
    __size_default = (512,512)

    def __init__(self, size=__size_default):
        self.__size_default = size

    def load_images(self, x):
        image_string = tf.io.read_file(x)
        image_decode = tf.image.decode_png(image_string, channels=3)
        image_norm = tf.image.resize(image_decode, size=self.__size_default)
        image_norm = image_norm / 255.0
        image_final = tf.cast(image_norm, tf.float32)

        return image_final

    def load_labels(self, x):
        mask_string = tf.io.read_file(x)
        mask_decode = tf.image.decode_png(mask_string, channels=3)
        mask_norm = tf.image.resize(mask_decode, size=self.__size_default, method='nearest')
        mask_final = tf.cast(mask_norm, tf.uint8)

        return mask_final

