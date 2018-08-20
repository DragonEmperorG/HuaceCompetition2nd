# -*- coding: utf-8 -*-
"""
@author: LONG QIAN
"""
 
import os 
import sys
import csv
import configparser
import numpy as np
import tensorflow as tf 

# from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def make_dataset():
    cwd = os.getcwd()
    raw_data_path = os.path.join(cwd, 'data_processing','PredictionMAT')
    # 要生成的文件
    predition_writer = tf.python_io.TFRecordWriter(os.path.join(cwd, 'dataset', "ion_prediction.tfrecords"))
    csv_reader = csv.reader(open(os.path.join(cwd, 'data_processing', "PreExogenous.txt")))
    conf = configparser.ConfigParser()

    row_list = []
    for row in csv_reader:
        row.pop(0)
        row_list.append(row)
    
    ext_data_np = np.array(row_list, dtype=np.float)
    # print(ext_data_np.shape)
    
    input_time_steps = 24
    output_time_steps = 12
    iot_time_steps = input_time_steps + output_time_steps

    # 初始化图片序列处理的相关变量
    time_step_counter    = 0    # 时间戳记录变量
    nb_predition_samples = 0    # 预测集样本统计变量
    # total_time_step        = len(os.listdir(raw_data_path))    # 时间戳总量

    # 样本构建滑动窗口变量初始化
    iot_img_sequences      = []
    input_img_sequences    = []
    output_img_sequences   = []

    iot_time_sequences     = []  
    input_time_sequences   = []
    output_time_sequences  = []

    iot_ext_sequences      = []
    input_ext_sequences    = []

    for img_name in tqdm(os.listdir(raw_data_path), desc="TFRecordWriterProgress", unit="mat", ascii=True):
        # 每一个图片的地址
        img_path = os.path.join(raw_data_path, img_name)

        if(not os.path.exists(img_path)):
            continue

        # img = Image.open(img_path)
        img =loadmat(img_path)
        # 将图片转化为矩阵，并获取当前图片的时间戳
        img_raw_array = np.array(img['mat'])
        img_raw_array = img_raw_array.astype(np.float)
        img_raw_array = np.reshape(img_raw_array, (img_raw_array.shape[0], img_raw_array.shape[1], 1))
        img_time_step = int(os.path.splitext(img_name)[0])

        iot_img_sequences.append(img_raw_array)
        iot_time_sequences.append(img_time_step)
        iot_ext_sequences.append(ext_data_np[time_step_counter,:])

        time_step_counter = time_step_counter + 1

        # 当滑动窗口内的样本达到一个输入输出量的时候开始生成样本
        if (time_step_counter >= iot_time_steps):
            # 分离输入输出样本
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

            # example对象对input_img_sequences、input_ext_sequences和output_img_sequences等e数据进行封装
            # example = tf.train.Example(features=tf.train.Features(feature={
            #     'input_img_sequences'        : _bytes_feature([input_img_sequences_array.tobytes()]),
            #     'output_img_sequences'       : _bytes_feature([output_img_sequences_array.tobytes()]),
            #     'input_time_sequences'       : _int64_feature(input_time_sequences_array.flatten().tolist()),
            #     'output_time_sequences'      : _int64_feature(output_time_sequences_array.flatten().tolist()),
            #     'input_ext_sequences'        : _float_feature(input_ext_sequences_array.flatten().tolist()),
            #     'input_img_sequences_shape'  : _int64_feature(input_img_sequences_array.shape),
            #     'output_img_sequences_shape' : _int64_feature(output_img_sequences_array.shape),
            #     'input_ext_sequences_shape'  : _int64_feature(input_ext_sequences_array.shape),
            # }))
            example = tf.train.Example(features=tf.train.Features(feature={
                'input_img_sequences'        : _float_feature(input_img_sequences_array.flatten().tolist()),
                'output_img_sequences'       : _float_feature(output_img_sequences_array.flatten().tolist()),
                'input_time_sequences'       : _int64_feature(input_time_sequences_array),
                'output_time_sequences'      : _int64_feature(output_time_sequences_array),
                'input_ext_sequences'        : _float_feature(input_ext_sequences_array.flatten().tolist()),
                'input_img_sequences_shape'  : _int64_feature(input_img_sequences_array.shape),
                'output_img_sequences_shape' : _int64_feature(output_img_sequences_array.shape),
                'input_ext_sequences_shape'  : _int64_feature(input_ext_sequences_array.shape),
            }))

            nb_predition_samples = nb_predition_samples + 1
            predition_writer.write(example.SerializeToString())
            

            iot_img_sequences.pop(0)
            iot_time_sequences.pop(0)
            iot_ext_sequences.pop(0)
    
    predition_writer.close()

    # 统计回写数据集的相关参数
    conf["DatasetInfo"] = {        
        'nb_time_step'         : time_step_counter,
        'nb_prediction_samples' : nb_predition_samples,
    }
    with open(os.path.join(cwd, 'dataset', 'ion_dataset_prediction_info.ini'), 'w') as configfile:
        conf.write(configfile)


def main():
    make_dataset()

if  __name__ == '__main__':
    main()
