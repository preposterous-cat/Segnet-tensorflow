from app import app
from flask import render_template, request
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from app import image_process as improc
from app import metrics
from app import model_load as moload
from PIL import Image
from app import image_load as imload

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/test/segment', methods=['GET'])
def test_seg():
    return render_template('test_seg.html')


@app.route('/test/segment', methods=['POST'])
def process_self():
    img_file = request.files['ori_img']
    img_ext = img_file.filename.split('.')[-1]
    img_ext = img_ext.lower()
    if img_ext == 'png':
        img_path = r'app\static\img\upload\original\\' + img_file.filename
        img_file.save(img_path)

        img_load = imload.ImageLoad((512,512))
        func_load_image = img_load.load_images

        model = moload.ModelLoad('app\\50_lung_infec_seg_NEW.h5')

        pipeline = model.make_pipeline(img_path, func_load_image)
        img_pred = model.make_predict(pipeline)
        
        img_process = improc.ImageProcess()
        
        #img_pred = img_process.convertTo(img_pred, tf.shape(img_pred[0]), rgb=False)
        for image in pipeline.take(1):
            img_pred= img_process.convertTo(img_pred, tf.shape(image[0]), rgb=True)
            #img_erosion = img_process.doErode(img_pred)
        
        pred_path = r'app\static\img\upload\prediction\\' + img_file.filename
        ndarray = np.array(img_pred)
        img_erosion = img_process.doErode(ndarray)
        im = Image.fromarray(img_erosion)
        im.save(pred_path)
        
        return render_template('test_seg.html', filename=img_file.filename)
    else:
        return render_template('test_seg.html', err=True)

@app.route('/test/metrics', methods=['GET'])
def test_list():
    return render_template('test_metrics.html')

@app.route('/test/metrics', methods=['POST'])
def process_list():
    img_file = request.files.getlist('ori_img')
    metrics_list= []
    m_iou1=0
    m_iou2=0
    m_iou3=0
    m_miou=0
    m_time=0

    model = moload.ModelLoad('app\\50_lung_infec_seg_NEW.h5')
    img_load = imload.ImageLoad((512,512))
    func_pipeline = img_load.load_images
    img_process = improc.ImageProcess()

    for img in img_file:
        img_ext = img.filename.split('.')[-1]
        img_ext = img_ext.lower()
        if img_ext == 'png':
            img_path = r'app\static\img\upload\original\\' + img.filename
            img.save(img_path)

            pipeline = model.make_pipeline(img_path, func_pipeline)
            
            time_start = time.time()
            img_pred = model.make_predict(pipeline)
            time_end = time.time() - time_start
            img_pred = img_process.convertTo(img_pred, (512*512, 4), rgb=False)

            label_path = r'app\static\img\upload\label2\\' + img.filename
            label_img = img_load.load_labels(label_path)
            label_img = img_process.onehot_mask(label_img)

            img_metrics = metrics.Metrics(label_img, img_pred)

            iou1 = img_metrics.iou(label=0)
            iou2 = img_metrics.iou(label=1)
            iou3 = img_metrics.iou(label=2)
            miou = img_metrics.mean_iou()

            metrics_data = {
                'filename' : img.filename,
                'iou1' : iou1.numpy(),
                'iou2' : iou2.numpy(),
                'iou3' : iou3.numpy(),
                'miou' : miou.numpy(),
                'time' : time_end
            }

            metrics_list.append(metrics_data)
            m_iou1 += iou1
            m_iou2 += iou2
            m_iou3 += iou3
            m_miou += miou
            m_time += time_end
        else:
            return render_template('test_metrics.html', err=True)
    
    n_img = len(metrics_list)
    m_iou1 = m_iou1 / n_img
    m_iou2 = m_iou2 / n_img
    m_iou3 = m_iou3 / n_img
    m_miou = m_miou / n_img
    m_time = m_time / n_img
        
    return render_template(
        'test_metrics.html', 
        err=False, 
        filename=img.filename, 
        metrics=metrics_list,
        m_iou1=m_iou1.numpy(), 
        m_iou2=m_iou2.numpy(), 
        m_iou3=m_iou3.numpy(), 
        m_miou=m_miou.numpy(), 
        m_time=m_time
    )