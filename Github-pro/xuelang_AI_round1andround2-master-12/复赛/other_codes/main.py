# -*- coding: utf-8 -*-

"""
Created on Sun Aug  5 09:58:26 2018

@author: 飞的更高队代码

"""

import os
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
#import numpy as np
from keras.models import Sequential
from keras.initializers import he_normal
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing import image
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import xml.etree.cElementTree as et
import shutil
#加载keras模块
#from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.preprocessing.image import ImageDataGenerator

 #创建文件夹
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")
        
        
def resize_img_test(dir,savepath,height, width):
    i=0
    for file in os.listdir(dir):  
        img=cv2.imread(dir+"/"+file)
        res = cv2.resize(img,(height,width), interpolation = cv2.INTER_AREA)
        cv2.imwrite(savepath+"/"+file,res)
        i=i+1
def cv_imread(file_path):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1) 
    return cv_img   
    
def distort_color(image, color_ordering):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox):
    # 查看是否存在标注框。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
    # 随机的截取图片中一个块。
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=1)
    #bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        #tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    scipy.misc.imsave('../data/'+'zengqiang_xc.jpeg', distorted_image.eval())
    img=cv_imread('../data/'+'zengqiang_xc.jpeg')
    res = cv2.resize(img,(height,width), interpolation = cv2.INTER_AREA)
    return res
def crop_img_xc(dir,target_size,num,savepath,i):
    tf.reset_default_graph()
    with tf.Session() as sess:
        for dir2 in os.listdir(dir):  
            for file in os.listdir(dir+dir2+"/"):  
                if('xml'in file):
                    s=dir+dir2+"/"+file
                    tree=et.parse(s)
                    root=tree.getroot()
                    filename=root.find('filename').text
                    for Object in root.findall('object'):
                        bndbox=Object.find('bndbox')
                        xmin=bndbox.find('xmin').text
                        ymin=bndbox.find('ymin').text
                        xmax=bndbox.find('xmax').text
                        ymax=bndbox.find('ymax').text
                        image_raw_data = tf.gfile.FastGFile(dir+dir2+"/"+filename, 'rb').read()
                        img_data = tf.image.decode_jpeg(image_raw_data)       
                        img_data.set_shape([1920,2560,3])
                        a=img_data.get_shape()
                        box=tf.constant([[[int(ymin)/int(a[0]),int(xmin)/int(a[1]),int(ymax)/int(a[0]),int(xmax)/int(a[1])]]], dtype=tf.float32)
                        for j in range(num):
                             res = preprocess_for_train(img_data, target_size, target_size, box)
                             cv2.imwrite(savepath+"/zengqiang_xc_"+str(i)+".jpg",res)
                             i=i+1
                    img=cv_imread(dir+dir2+"/"+filename)
                    res2 = cv2.resize(img,(target_size,target_size), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(savepath+"/zengqiang_xc_"+str(i)+".jpg",res2)
                    i=i+1     
    return i
def crop_img_zc(dir,target_size,num,savepath,i):
    tf.reset_default_graph()
    with tf.Session() as sess:
        for file in os.listdir(dir):
            filename=file
            image_raw_data = tf.gfile.FastGFile(dir+filename, 'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            img_data.set_shape([1920,2560,3])
            #a=img_data.get_shape()
            box=None
            for j in range(num):
                res = preprocess_for_train(img_data, target_size, target_size, box)
                cv2.imwrite(savepath+"/zengqiang_zc_"+str(i)+".jpg",res)
                i=i+1
            img=cv_imread(dir+filename)
            res2 = cv2.resize(img,(target_size,target_size), interpolation = cv2.INTER_AREA)
            cv2.imwrite(savepath+"/zengqiang_zc_"+str(i)+".jpg",res2)
            i=i+1
    return i
            
  
def get_files(file_dir):
    zc = []
    label_zc = [] 
    xc = []
    label_xc = []
    
    for file in os.listdir(file_dir+'/label_0'):
        zc.append(file_dir +'/label_0'+'/'+ file)
        label_zc.append(0)
    for file in os.listdir(file_dir+'/label_1'):
        xc.append(file_dir +'/label_1'+'/'+file)
        label_xc.append(1)
#把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((zc, xc))
    label_list = np.hstack((label_zc, label_xc))
 
#利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
 
#从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]                                                   
    return image_list,label_list
#返回两个list 分别为图片文件名及其标签  顺序已被打乱
            
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
 
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        
        
        
# build model
def VGG_13_model():
    model = Sequential()
    weight_decay = 0.0001
    dropout=0.5
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block1_conv1', input_shape=(499,499,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block3_conv3'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block3_conv4'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block4_conv3'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block4_conv4'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    
    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block5_conv3'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block5_conv4'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    # model modification for cifar-10
    #model.add(Flatten(name='flatten'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1080, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                kernel_initializer=he_normal(), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1080, kernel_regularizer=keras.regularizers.l2(weight_decay),
                kernel_initializer=he_normal(), name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(weight_decay),
                kernel_initializer=he_normal(), name='fc3'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model
    
    #G=4
    #if G <= 1:
    #    print("[INFO] training with 1 GPU...")
    #    parallel_model = VGG19_model()
    #
    ## otherwise, we are compiling using multiple GPUs
    #else:
    #    print("[INFO] training with {} GPUs...".format(G))
    #    # we'll store a copy of the model on *every* GPU and then combine
    #    # the results from the gradient updates on the CPU
    #    with tf.device("/cpu:0"):
    #        # initialize the model
    #        model = VGG19_model()
    #        # make the model parallel(if you have more than 2 GPU)
    #    parallel_model = multi_gpu_model(model, gpus=G)
    #    parallel_model.__setattr__('callback_model', model)
	

if __name__=="__main__":
    #创建所有所需的文件夹
    file = "../data/train_data"
    mkdir(file)          
    file2= "../data/test_data"
    mkdir(file2)         
    file3= "../data/resize_test"
    mkdir(file3) 
    file4= "../data/all_data/label_0"
    mkdir(file4) 
    file5= "../data/all_data/label_1"
    mkdir(file5) 
    file6= "../data/train_data/label_0"
    mkdir(file6) 
    file7= "../data/train_data/label_1"
    mkdir(file7) 
    file8= "../data/test_data/label_0"
    mkdir(file8) 
    file9= "../data/test_data/label_1"
    mkdir(file9) 
    file10= "../data/VGG13_models"
    mkdir(file10) 
    #对所有图片进行预处理
    i0=0
    i1=crop_img_xc('../data/xuelang_round1_train_part1_20180628/',499,3,'../data/all_data/label_1',i0)
    i2=crop_img_xc('../data/xuelang_round1_train_part2_20180705/',499,3,'../data/all_data/label_1',i1)
    i3=crop_img_xc('../data/xuelang_round1_train_part3_20180709/',499,3,'../data/all_data/label_1',i2)
    
    i4=0
    i5=crop_img_zc('../data/xuelang_round1_train_part1_20180628/正常/',499,1,'../data/all_data/label_0',i4)
    i6=crop_img_zc('../data/xuelang_round1_train_part2_20180705/正常/',499,1,'../data/all_data/label_0',i5)
    i7=crop_img_zc('../data/xuelang_round1_�vk3��L4p�4���4�]�4�8�3�`�4�4-l�4�5�Ǻ3x�D4��4�:h4��*3��3(G�4M�4�_4F.�3�@_4�!�4��n5(u6��=4xsc5���5�4�5�86�ʃ5=�5Y�Y62-�54�5|C�5س5�Ԝ5*�4�ݐ5��5�6	�5�m�4�C5��5�2a5lSO5�6�4Dۅ56l�5s��5��*5�J�5;ʸ5���5��W5\�J6�-�5	�5K�68��5N\�5y)�5ۜ�5�DC65|�4hh�5��5LIc6�o�5��`5H��4�a�5{��5i54��X5��5�C6�\5���4�6�5@��5���4��5y5�+�4]�5��5�le5eq�5'06�Va5τ5��5���5�r�5���5�ۣ4���50X�4�r�6�I�5��Q5�5�5Hr:5��L5Jv�5�5�T�5�	6w6��5��H5/�5x(P6�2�4��N5v4ɖF5�mq5�l6�e6e$5
�6`V�5z��5쁷5��5*�4�[6^y�5,6���6�X5�<5A�6T��6&�86#n5��>6'5h^~5��6�:.5�q6�05%��6��V6�C;6��4�%C5)�5�6�)�5F_�5Ѳ�5-s5�aZ5���4�ˋ5���5�3y5��(5�j6�M�4��5�=-6�#�5<Y�5(4�Г56e05\�4�a%5Z��5�� 5d�5�۵5��5��6F�6hz?5�_�4�|6�	5Z�&6��I6 vd5*B�3�r�3w�A6�V4��5�.�5z#�5��5I;G6�?�5��E5�C14+@ 5�0�4��v5�m�5��6u��6a�5^@:6>��5���4#q6\j�5���5(�4*$5�5L��5H�6�X�5��5J��5�5��5���5���5'5��5Ӧ6#�5�1�5��-6�K5�WE6�@%5"R�5��H4�|�5*P6��'5^�k5*�4���4�;	60�,6ݎ�5���5H��4	�5:�596.�6�E�4>l6�<55�g�5�;�4`��4O )6��I5Z<
6
ps5���5�n�5��!5�p?6iEL5q�5��L629c6�=�4j;5w��5d%�5��5���4l�5)2D6h��4�0\5-@3��<4�y+3>�3�T�4��^4HN�5z�.4��`4Ѧ�4�c5104���4��2���43�3���4� �4��;4�94���3N/A4��24�Z�3�B4(?&4�g!48ш3OSd4��4Ym�3
[�4��50X5�VE5��\5�	�4��4���3�;�4�tG5�w.46a�4H��4#5��p4h�4FB3��4�V�3W�o2��4� y4r��4�G4�35b��4�32�4I�3�ҿ3y%�3�ş3R�4��]4�?5�Ww4���3���3���4�>5�0�4��3I'�4�/4H�t5� �3��@4�ܑ4��3��3�d�4��:4�#�4zd�4�4���4kH]4���4Ғ5���35�4�(3�B4�Q�4N)5l(V4�054��O5���4`+4Nmj5�7�4i,3W[42�3�)�5n�U4XT+3~��3M��4�{�40�4��R4���4T�43Q�4l5�C�3X��4H=4��5�T�4�I4t�+3?��3�j�3N�n5�{�4$� 4�0�4T��3*�r4<Q4�#�3���4'�3��3��L5J%�3��4t̻4��f4v�b4��83�4C�94�$'4M�3��3�_4�r�4U!�4��4C=�4���4�s5`5�3���2��4�W4:��4��4P@<4;І3n��2Vm5���3A�3��4u�g4�� 5��4E=5��b4���3�� 4���3O�4f4{�446aJ4.G4�ns4P�$3�5��4���3��3x�A3w�!4*y4���3\�94��J4���4�4Vc4��3�(5��3��05O� 5�l4���3WbU5�Z�3�E5@�3t��3��4T�4���3��4�0�4�ri3��4�	5L�N4���4���4�
4��>4z%4�X4ht�4���3Z�5�w 4p��4��3i�G4�A�4�14N 5��4�4���4i�z4Az�4�=�3�Zz46y�5x�4ǵ�3���3�~�4�O%5��3U�h4քL5k
�4e�4���5�Z4��5i4�W�49�'5�\5AT�5b�L5�F5�C�4u'35�4�$�5ĦS4�A5X-�4�q�41L5�C�4�$5�?5QS>5���4��o45�!5
c-4T f5E�k4O15�>5��
5��\5��6���5��a5��5�_5� 5�Я5h'�5�5��4/�5�W5�[�50�@5ȼ�4I�4��5�24�!�3�4�E*5��*5���4@�4@��4g]G5:G_4٤�5���4�ɾ4��
5��4@�4���4К�5�a5�Ǯ4�`�4f�?5l\5�R"5�/�4�35o�94���5�k�3r�5#�N5ג�4�$p4��^5@��4~Ij5��b5�C�5�p5M	�5���5N�50��4�(	5�^3�25��5�]583�57,G5�j�5o6�4/�D5�2�4ʄ.5.U4��(5���4.�4�5��4�4��6԰�5� �4��-5J�5��4m��4 ɧ5.ʨ57� 6���4�{5���4��K5ܛ�3�e�4�4�
.6�%5}4 P5��49j�4�4� Y4��h5e�/4���4�0�5�4�,�4���5u5j>.5�ʺ4MB4-x�4�Dg4 mh4��@4g�'5D��4Rm�4l);5��4��5�.5��4]I�3�z95���4�C6$<F5
�E4.��4 �3�I�5�� 4��|4�:�4t�y4I1}5�(�5%��5@ޏ4�ա3� 5Qe�4Þ45��;5q`�5�#�6���4a.5]�4A�4�)N5�K\4%��4�B4//S4S��5��4]T5.v�5w��4·�5�D54�4	5�v5wl4l��4�ם5j��4�3�4oÃ5||[48A�5��4)Z5^�4��L57!5�^5���4�JN4���4�@5��!5��S4>��5�v*4[-�4d9�5;b5���5���4E�5N��4XWS5��-42��4��5&�t5�N�4��4��M57��5��41g6�px4���59��5�(5��49�4��4�=5��4y��4��4��4�E5�f5��3��+5�>x4h�84z\'5���4f�4Z5�5��q4��5]x�3g85m��3z*@5�
�4I�@4��w4e5���4͑�4��5Fv5:*
4��4M�$4�35�(�3�AY4��25�� 5�T�4N�5D��5�?5��5v�4���4��85�m�4zL�5�3�4b�5~��4�E5�h�4��
5�+4iw/5�aN4R4+4���4�"�4`XU4`z4s�^4o�#5���4C��3��+5��84Oα4F��4�^5�]5�׋4\�5�<5���4�^4�5��5�J�4�#�3�Q5){4���5��3�5$5}4S�h4��	4��5E;65��X4�5�yU5���5�ͫ4��6���5�l15!�4ih~3e9�3��4l�5�!5�O4Enm5��5=��4�f�5��4.C4���4��T4�&5_~�4_o%5E4w/05&66w'�4��4���4B�y4�� 5��5���4:e�5��r4���4��5	��4e�m3��D5.yP4��5�S4�4 nP5T�4��4�AH4���4�R5=��4���3n�$6F��3�ռ4���5�U4��4�3�0535@�{5;�B4�a4;��4�,5�A5~e>4}�4�b�5L�5�(4��4���4z	�3���4m��5[�?4��u4[�2�+A5�]�3]�4��;4]5j�5�m6g �4~>�4l�T3�w05|[4��95��W4�@5M<6,��4b�_5=�5{4	o"6�_4{U�3Ή)4�$4���4V��4���4Y�
5,I�4�S5�Ey4��V4��4�^5��3c^�4��5d�4�A�5�/�5O�,4���5ɕ�4�4sX�3��4�9�4
��4M��4�\�3)]Q4�5<��4���4��4QDQ4��5�5�[�4��N6�R�4��5`04�F5�ǂ3,�d4�35iw�4̬5�4fnU5�4b�5E��5}ғ4<�C4�0�5\��4��4�?4�l�5<_�4��b5WÐ4��5��?52e�5�
6�{�4�R 6;�W5'�(5�'6��5C5Yo�6\
6�6:�5ha55�A6�75j=�5�E�4l�6�ܮ5��5tM5�%
6�	$6'ą5"�3