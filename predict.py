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
# from PIL import Image
from scipy.io import savemat
from model.params import Params
from model.tec_pre_net import tec_pre_net


def parse_data(serialized_example):
    #return input_img_sequences and output_img_sequences
    features = tf.parse_single_example(
        serialized_example,
        features={
            'input_img_sequences'        : tf.FixedLenFeature([Params.input_time_steps*Params.map_rows*Params.map_cols], tf.float32),
            'output_img_sequences'       : tf.FixedLenFeature([Params.output_time_steps*Params.map_rows*Params.map_cols], tf.float32),
            'output_time_sequences'      : tf.FixedLenFeature([Params.output_time_steps], tf.int64),
            'input_ext_sequences'        : tf.FixedLenFeature([Params.input_time_steps*Params.external_dim], tf.float32),
            # 'input_img_sequences'        : tf.FixedLenFeature([71736], tf.float32),
            # 'output_img_sequences'       : tf.FixedLenFeature([35868], tf.float32),
            # 'output_time_sequences'      : tf.FixedLenFeature([12], tf.int64),
            # 'input_ext_sequences'        : tf.FixedLenFeature([120], tf.float32),
            'input_img_sequences_shape'  : tf.FixedLenFeature([4], tf.int64),
            'output_img_sequences_shape' : tf.FixedLenFeature([4], tf.int64),
            'input_ext_sequences_shape'  : tf.FixedLenFeature([2], tf.int64),
        }
    )
    
    output_time_sequences      = tf.cast(features['output_time_sequences'], tf.int32)
    input_img_sequences_shape  = tf.cast(features['input_img_sequences_shape'], tf.int32)
    output_img_sequences_shape = tf.cast(features['output_img_sequences_shape'], tf.int32)
    input_ext_sequences_shape  = tf.cast(features['input_ext_sequences_shape'], tf.int32)

    input_img_sequences = tf.reshape(features['input_img_sequences'], input_img_sequences_shape)
    output_img_sequences = tf.reshape(features['output_img_sequences'], output_img_sequences_shape)
    input_ext_sequences = tf.reshape(features['input_ext_sequences'], input_ext_sequences_shape)

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
    config.read(os.path.join(cwd, 'dataset', "ion_dataset_prediction_info.ini"))

    img_rows              = Params.map_rows
    img_cols              = Params.map_cols
    input_time_steps      = Params.input_time_steps
    output_time_steps     = Params.output_time_steps
    external_dim          = Params.external_dim
    # nb_prediction_samples  = 12
    nb_prediction_samples  = config.getint('DatasetInfo', 'nb_prediction_samples')

    load_weights_path     = os.path.join(cwd, 'checkpoint', '20180815001811', "TEC_PRE_NET_MODEL_WEIGHTS.01-200.7376-0.59353.hdf5")
    prediction_save_path  = os.path.join(cwd, 'prediction', datetime.now().strftime('%Y%m%d%H%M%S'))
    try:
        os.makedirs(prediction_save_path)
    except:
        pass

    # prediction_dataset          = load_data(os.path.join(cwd, 'dataset','ion_prediction.tfrecords'))
    prediction_dataset          = load_data(os.path.join(cwd, 'dataset','ion_prediction.tfrecords'))
    prediction_dataset_iterator = prediction_dataset.make_initializable_iterator()
    prediction_dataset_iterator_next_element = prediction_dataset_iterator.get_next()

    input_img_sequences_prediction  = []
    input_ext_sequences_prediction  = []
    output_img_sequences_prediction = []
    output_time_sequences_prediction = []

    #开始一个会话
    # with tf.Session(config=tfconfig) as sess:
    with tf.Session() as sess:  
        sess.run(prediction_dataset_iterator.initializer)

        try:
            while True:
                input_img_sequences, input_ext_sequences, output_img_sequences, output_time_sequences = sess.run(prediction_dataset_iterator_next_element)
                input_img_sequences_prediction.append(input_img_sequences)
                input_ext_sequences_prediction.append(input_ext_sequences)
                output_img_sequences_prediction.append(output_img_sequences)
                output_time_sequences_prediction.append(output_time_sequences)

        except tf.errors.OutOfRangeError:
            print("prediction dataset constructed ...")

    print("Start model construction ...")
    model = tec_pre_net((img_rows, img_cols), input_time_steps, output_time_steps, external_dim)
    model.load_weights(load_weights_path, by_name=False)
    print("Model constructed ...")

    for prediction_samples_counter in tqdm(range(nb_prediction_samples), desc="IONPredictionProgress", unit="sequences", ascii=True):
        input_img_sequences_predict_fit_x = np.expand_dims(input_img_sequences_prediction[prediction_samples_counter], axis=0)
        input_ext_sequences_predict_fit_x = np.expand_dims(input_ext_sequences_prediction[prediction_samples_counter], axis=0)

        output_img_sequences_predict_raw = model.predict([input_img_sequences_predict_fit_x, input_ext_sequences_predict_fit_x])
        # output_img_sequences_predict = output_img_sequences_predict_raw.astype(int)
        output_img_sequences_predict = output_img_sequences_predict_raw

        output_img_sequences_prediction_concatenate    = np.reshape(output_img_sequences_prediction[prediction_samples_counter], (output_time_steps*img_rows, img_cols, 1))
        output_img_sequences_predict_concatenate = np.reshape(output_img_sequences_predict[0], (output_time_steps*img_rows, img_cols, 1))

        output_img_sequences_concatenated = np.concatenate((output_img_sequences_prediction_concatenate, output_img_sequences_predict_concatenate), axis=1)
        
        output_img_sequences_concatenated_2D = np.reshape(output_img_sequences_concatenated, (output_time_steps*img_rows, img_cols+img_cols))
        output_img_sequences_matlab_name = str(output_time_sequences_prediction[prediction_samples_counter][0]) + '-' + str(output_time_sequences_prediction[prediction_samples_counter][-1]) + ".mat"
        savemat(
            os.path.join(prediction_save_path, output_img_sequences_matlab_name),
            {
                'output_img_sequences_concatenated': output_img_sequences_concatenated_2D
            }
        )

        # output_img_sequences_concatenated_2D_uint8 = output_img_sequences_concatenated_2D.astype(np.uint8)
        # output_img_sequences_visual = Image.fromarray(output_img_sequences_concatenated_2D_uint8, mode="L")
        # output_img_sequences_visual_name = str(output_time_sequences_prediction[prediction_samples_counter][0]) + '-' + str(output_time_sequences_prediction[prediction_samples_counter][-1]) + ".jpeg"
        # output_img_sequences_visual.save(os.path.join(prediction_save_path, output_img_sequences_visual_name))
