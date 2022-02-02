import tensorflow as tf
from keras.layers import Layer
from keras import backend as K

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        return [output, argmax]

    def get_config(self):
      config = super().get_config().copy()
      config.update({
          'padding': self.padding,
          'pool_size': self.pool_size,
          'strides': self.strides
      })
      return config
    

class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope('unpooling2D'):
            mask = tf.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            output_shape_ = tf.shape(output, out_type=tf.int32)
            #  calculation new shape
            output_shape = (input_shape[0], output_shape_[1], output_shape_[2], input_shape[3])
            
            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask, dtype='int32')
            batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range
            
            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
            values = tf.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            set_input_shape = updates.get_shape()
            prev_shape = output.get_shape()

            set_output_shape = [set_input_shape[0], prev_shape[1], prev_shape[2], set_input_shape[3]]
            ret.set_shape(set_output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        output_shape = [mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]]
        return tuple(output_shape)
  
    def get_config(self):
      config = super().get_config().copy()
      config.update({
          'size': self.size,
      })
      return config