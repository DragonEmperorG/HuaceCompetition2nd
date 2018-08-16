# -*- coding: utf-8 -*-
"""
@author: LONG QIAN
"""


import os 
import configparser
import numpy as np
import tensorflow as tf 

from datetime import datetime
from tqdm import tqdm
from PIL import Image
from model.params import Params
from model.tec_pre_net import tec_pre_net


def parse_data(serialized_example):
    #return input_img_sequences and output_img_sequences
    features = tf.parse_single_example(
        serialized_example,
        features={
            'input_img_sequences'        : tf.FixedLenFeature([], tf.string),
            'output_img_sequences'       : tf.FixedLenFeature([], tf.string),
            'output_time_sequences'      : tf.FixedLenFeature([Params.output_time_steps], tf.int64),
            'input_ext_sequences'        : tf.FixedLenFeature([180], tf.float32),
            'input_img_sequences_shape'  : tf.FixedLenFeature([4], tf.int64),
            'output_img_sequences_shape' : tf.FixedLenFeature([4], tf.int64),
            'input_ext_sequences_shape'  : tf.FixedLenFeature([2], tf.int64),
        }
    )

    input_img_sequences = tf.decode_raw(features['input_img_sequences'], tf.uint8)
    output_img_sequences = tf.decode_raw(features['output_img_sequences'], tf.uint8)

    output_time_sequences      = tf.cast(features['output_time_sequences'], tf.int32)
    input_img_sequences_shape  = tf.cast(features['input_img_sequences_shape'], tf.int32)
    output_img_sequences_shape = tf.cast(features['output_img_sequences_shape'], tf.int32)
    input_ext_sequences_shape  = tf.cast(features['input_ext_sequences_shape'], tf.int32)

    input_img_sequences = tf.reshape(input_img_sequences, input_img_sequences_shape)
    output_img_sequences = tf.reshape(output_img_sequences, output_img_sequences_shape)
    input_ext_sequences = tf.reshape(features['input_ext_sequences'], input_ext_sequences_shape)
    #throw input_img_sequences tensor
    # input_img_sequences = tf.cast(input_img_sequences, tf.int32)
    #throw output_img_sequences tensor
    # output_img_sequences = tf.cast(output_img_sequences, tf.int32)

    return input_img_sequences, input_ext_sequences, output_img_sequences, output_time_sequences


def load_data(filename):
    # create a queue
    # filename_queue = tf.train.string_input_producer([filename])

    # reader = tf.TFRecordReader()
    # return file_name and file
    # _, serialized_example = reader.read(filename_queue)
    dataset = tf.data.TFRecordDataset(filename)
    return dataset.map(parse_data)

    
if __name__ == '__main__':
    cwd = os.getcwd()
    config = configparser.ConfigParser()
    config.read(os.path.join(cwd, 'dataset', "ion_dataset_info.ini"))

    img_rows              = Params.map_rows
    img_cols              = Params.map_cols
    input_time_steps      = Params.input_time_steps
    output_time_steps     = Params.output_time_steps
    nb_test_samples       = config.getint('DatasetInfo', 'nb_test_samples')

    load_weights_path     = os.path.join(cwd, 'checkpoint', '20180815200639', "TEC_PRE_NET_MODEL_WEIGHTS.03-0.01570.hdf5")
    prediction_save_path  = os.path.join(cwd, 'prediction', datetime.now().strftime('%Y%m%d%H%M%S'))
    try:
        os.makedirs(prediction_save_path)
    except:
        pass

    test_dataset          = load_data(os.path.join(cwd, 'dataset','ion_test.tfrecords'))
    test_dataset_iterator = test_dataset.make_initializable_iterator()
    test_dataset_iterator_next_element = test_dataset_iterator.get_next()

    input_img_sequences_test  = []
    input_ext_sequences_test  = []
    output_img_sequences_test = []
    output_time_sequences_test = []

    #开始一个会话
    # with tf.Session(config=tfconfig) as sess:
    with tf.Session() as sess:  
        sess.run(test_dataset_iterator.initializer)

        try:
            while True:
                input_img_sequences, input_ext_sequences, output_img_sequences, output_time_sequences = sess.run(test_dataset_iterator_next_element)
                input_img_sequences_test.append(input_img_sequences)
                input_ext_sequences_test.append(input_ext_sequences)
                output_img_sequences_test.append(output_img_sequences)
                output_time_sequences_test.append(output_time_sequences)

        except tf.errors.OutOfRangeError:
            print("Test dataset constructed ...")

    print("Start model construction ...")
    model = tec_pre_net((img_rows, img_cols))
    model.load_weights(load_weights_path, by_name=False)
    print("Model constructed ...")

    for test_samples_counter in tqdm(range(nb_test_samples), desc="IONPredictionProgress", unit="sequences", ascii=True):
        input_img_sequences_predict_fit_x = np.expand_dims(input_img_sequences_test[test_samples_counter], axis=0)
        input_ext_sequences_predict_fit_x = np.expand_dims(input_ext_sequences_test[test_samples_counter], axis=0)

        output_img_sequences_predict_raw = model.predict([input_img_sequences_predict_fit_x, input_ext_sequences_predict_fit_x])
        output_img_sequences_predict = output_img_sequences_predict_raw.astype(int)

        output_img_sequences_test_concatenate    = np.reshape(output_img_sequences_test[test_samples_counter], (output_time_steps*img_rows, img_cols, 1))
        output_img_sequences_predict_concatenate = np.reshape(output_img_sequences_predict[0], (output_time_steps*img_rows, img_cols, 1))

        output_img_sequences_concatenated = np.concatenate((output_img_sequences_test_concatenate, output_img_sequences_predict_concatenate), axis=1)
        
        output_img_sequences_concatenated_2D = np.reshape(output_img_sequences_concatenated, (output_time_steps*img_rows, img_cols+img_cols))
        output_img_sequences_concatenated_2D_uint8 = output_img_sequences_concatenated_2D.astype(np.uint8)
        output_img_sequences_visual = Image.fromarray(output_img_sequences_concatenated_2D_uint8, mode="L")

        output_img_sequences_visual_name = str(output_time_sequences_test[test_samples_counter][0]) + '-' + str(output_time_sequences_test[test_samples_counter][-1]) + ".jpeg"
        output_img_sequences_visual.save(os.path.join(prediction_save_path, output_img_sequences_visual_name))
