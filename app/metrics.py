from keras import backend as K

class Metrics:
    __ytrue=0
    __ypred=0
    def __init__(self, ytrue, ypred):
        self.__ytrue=ytrue
        self.__ypred=ypred
        
    def iou(self, label: int):
        # Mengubah input menjadi scalar dan hanya mengambil value sesuai
        # dengan label
        y_true = K.cast(K.equal(K.argmax(self.__ytrue), label), K.floatx())
        y_pred = K.cast(K.equal(K.argmax(self.__ypred), label), K.floatx())
        # Menghitung Intersection
        intersection = K.sum(y_true * y_pred)
        # Menghitung Union
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        # Apabila Union bernilai 0, maka akan diganti dengan 1 untuk menghindari
        # pembagian 0, jika tidak maka akan dilakukan operasi intersection / union
        return K.switch(K.equal(union, 0), 1.0, intersection / union)

    def mean_iou (self):
        label1 = self.iou(label=0)
        label2 = self.iou(label=1)
        label3 = self.iou(label=2)
        result = (label1 + label2 + label3) / 3
        return result

    def precision(self, label):
        mask = K.cast(K.equal(K.argmax(self.__ytrue), label), K.floatx())
        pred = K.cast(K.equal(K.argmax(self.__ypred), label), K.floatx())
        tp = K.sum(mask * pred)
        tp_fp = K.sum(mask)
        precision = tp / tp_fp
        return precision
    
    def mean_precision (self):
        label1 = self.precision(self.__ytrue, self.ypred, 0)
        label2 = self.precision(self.__ytrue, self.ypred, 1)
        label3 = self.precision(self.__ytrue, self.ypred, 2)
        result = (label1 + label2 + label3) / 3
        return result