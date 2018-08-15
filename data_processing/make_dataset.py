# -*- coding: utf-8 -*-
"""
@author: LONG QIAN
"""
 
import os 
import csv
import configparser
import numpy as np
import tensorflow as tf 

from PIL import Image
from tqdm import tqdm
#from model.params import Params


if  __name__ == '__main__':

    cwd = os.getcwd()
    np.random.seed(0)
    raw_data_path = os.path.join(cwd, 'data_processing','JPG')
    # 要生成的文件
    training_writer = tf.python_io.TFRecordWriter(os.path.join(cwd, 'dataset', "ion_training.tfrecords"))
    validation_writer = tf.python_io.TFRecordWriter(os.path.join(cwd, 'dataset', "ion_validation.tfrecords"))
    test_writer = tf.python_io.TFRecordWriter(os.path.join(cwd, 'dataset', "ion_test.tfrecords"))
    csv_reader = csv.reader(open(os.path.join(cwd, 'data_processing', "exogenous.txt")))
    conf = configparser.ConfigParser()

    row_list = []
    for row in csv_reader:
        row.pop(0)
        row_list.append(row)
    
    ext_data_np = np.array(row_list, dtype=np.float)

    # print(ext_data_np.shape)
    
    input_time_steps = 36
    output_time_steps = 12
    iot_time_steps = input_time_steps + output_time_steps

    time_step_counter      = 0
    nb_train_samples       = 0
    nb_validation_samples  = 0
    nb_test_samples        = 0
    total_time_step        = len(os.listdir(raw_data_path))
    nb_samples             = 0

    iot_img_sequences      = []
    input_img_sequences    = []
    output_img_sequences   = []

    iot_time_sequences     = []  
    input_time_sequences   = []
    output_time_sequences  = []

    iot_ext_sequences      = []
    input_ext_sequences    = []
    output_ext_sequences   = []

    for img_name in tqdm(os.listdir(raw_data_path), desc="TFRecordWriterProgress", unit="image", ascii=True):
        # 每一个图片的地址
        img_path = os.path.join(raw_data_path, img_name)

        if(not os.path.exists(img_path)):
            continue

        img = Image.open(img_path)
        # 将图片转化为矩阵
        img_raw_array = np.array(img)
        img_raw_array = np.reshape(img_raw_array, (img_raw_array.shape[0], img_raw_array.shape[1], 1))
        img_time_step = int(os.path.splitext(img_name)[0])

        iot_img_sequences.append(img_raw_array)
        iot_time_sequences.append(img_time_step)
        iot_ext_sequences.append(ext_data_np[time_step_counter,:])

        time_step_counter = time_step_counter + 1

        if (time_step_counter >= iot_time_steps):
            nb_samples = nb_samples + 1

            input_img_sequences   = iot_img_sequences[:input_time_steps]
            output_img_sequences  = iot_img_sequences[input_time_steps:]
            input_time_sequences  = iot_time_sequences[:input_time_steps]
            output_time_sequences = iot_time_sequences[input_time_steps:]
            input_ext_sequences   = iot_ext_sequences[:input_time_steps]

            input_img_sequences_array   = np.array(input_img_sequences)
            output_img_sequences_array  = np.array(output_img_sequences)
            input_time_sequences_array  = np.array(input_time_sequences)
            output_time_sequences_array = np.array(output_time_sequences)
            input_ext_sequences_array   = np.array(input_ext_sequences)

            # print(input_img_sequences_array.shape)
            # print(input_ext_sequences_array.shape)

            #example对象对label和image数据进行封装
            example = tf.train.Example(features=tf.train.Features(feature={
                'input_img_sequences'        : tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_img_sequences_array.tobytes()])),
                'output_img_sequences'       : tf.train.Feature(bytes_list=tf.train.BytesList(value=[output_img_sequences_array.tobytes()])),
                'input_time_sequences'       : tf.train.Feature(int64_list=tf.train.Int64List(value=input_time_sequences_array.flatten().tolist())),
                'output_time_sequences'      : tf.train.Feature(int64_list=tf.train.Int64List(value=output_time_sequences_array.flatten().tolist())),
                'input_ext_sequences'        : tf.train.Feature(float_list=tf.train.FloatList(value=input_ext_sequences_array.flatten().tolist())),
                'input_img_sequences_shape'  : tf.train.Feature(int64_list=tf.train.Int64List(value=input_img_sequences_array.shape)),
                'output_img_sequences_shape' : tf.train.Feature(int64_list=tf.train.Int64List(value=output_img_sequences_array.shape)),
                'input_ext_sequences_shape'  : tf.train.Feature(int64_list=tf.train.Int64List(value=input_ext_sequences_array.shape)),
            }))

            if (np.random.randint(10) <= 7):
                #序列化为字符串
                nb_train_samples = nb_train_samples + 1
                training_writer.write(example.SerializeToString())

            elif (np.random.randint(10) == 8):
                nb_validation_samples = nb_validation_samples + 1
                validation_writer.write(example.SerializeToString())

            else:
                nb_test_samples = nb_test_samples + 1
                test_writer.write(example.SerializeToString())
            

            iot_img_sequences.pop(0)
            iot_time_sequences.pop(0)
            iot_ext_sequences.pop(0)
    
    training_writer.close()
    validation_writer.close()
    test_writer.close()

    conf["DatasetInfo"] = {        
        'nb_time_step'          : time_step_counter,
        'nb_samples'            : nb_samples,
        'nb_train_samples'      : nb_train_samples,
        'nb_validation_samples' : nb_validation_samples,
        'nb_test_samples'       : nb_test_samples,
    }
    with open(os.path.join(cwd, 'dataset', 'ion_dataset_info.ini'), 'w') as configfile:
        conf.write(configfile)
