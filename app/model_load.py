from app import custom_layer as cl
from app import metrics
import tensorflow as tf


class ModelLoad:
    __path_model=""

    def __init__(self, path):
        self.__path_model=path
        custom_objects = {"MaxPoolingWithArgmax2D": cl.MaxPoolingWithArgmax2D, "MaxUnpooling2D": cl.MaxUnpooling2D, "mean_iou" : metrics.Metrics.mean_iou}
        with tf.keras.utils.custom_object_scope(custom_objects):
            self.segnet = tf.keras.models.load_model(self.__path_model)
    
    def make_pipeline(self, path, func):
        img_dataset = tf.data.Dataset.from_tensor_slices([path])
        img_mload = img_dataset.map(func)
        img_batch = img_mload.batch(1)

        return img_batch
    
    def make_predict(self, img):
        pred_img = self.segnet.predict(img)

        return pred_img


#'app\\50_lung_infec_seg_NEW.h5'