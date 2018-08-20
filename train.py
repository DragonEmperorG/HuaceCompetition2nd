# -*- coding: utf-8 -*-
"""
@author: LONG QIAN
"""
from __future__ import print_function

import os 
import configparser
import numpy as np
import tensorflow as tf
import sys
# import keras.backend.tensorflow_backend as KTF
# tfconfig = tf.ConfigProto()
# tfconfig.gpu_options.allow_growth = True
# session = tf.Session(config=tfconfig)
# GPU 显存自动调用
# KTF.set_session(session)     


from datetime import datetime
from keras import backend as K
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, TensorBoard
from model.params import Params
from model.tec_pre_net import tec_pre_net

"""
    MinMaxNormalization
"""
class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        self._min = sys.float_info.max
        self._max = sys.float_info.min
        pass

    def fit(self, X):
        self._min = min(X.min(), self._min)
        self._max = max(X.max(), self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def parse_data(serialized_example):
    #return input_img_sequences and output_img_sequences
    features = tf.parse_single_example(
        serialized_example,
        features={
            'input_img_sequences'        : tf.FixedLenFeature([71736], tf.float32),
            'output_img_sequences'       : tf.FixedLenFeature([35868], tf.float32),
            'input_ext_sequences'        : tf.FixedLenFeature([120], tf.float32),
            'input_img_sequences_shape'  : tf.FixedLenFeature([4], tf.int64),
            'output_img_sequences_shape' : tf.FixedLenFeature([4], tf.int64),
            'input_ext_sequences_shape'  : tf.FixedLenFeature([2], tf.int64),
        }
    )

    input_img_sequences_shape  = tf.cast(features['input_img_sequences_shape'], tf.int32)
    output_img_sequences_shape = tf.cast(features['output_img_sequences_shape'], tf.int32)
    input_ext_sequences_shape  = tf.cast(features['input_ext_sequences_shape'], tf.int32)

    input_img_sequences = tf.reshape(features['input_img_sequences'], input_img_sequences_shape)
    output_img_sequences = tf.reshape(features['output_img_sequences'], output_img_sequences_shape)
    input_ext_sequences = tf.reshape(features['input_ext_sequences'], input_ext_sequences_shape)
    #throw input_img_sequences tensor
    # input_img_sequences = tf.cast(input_img_sequences, tf.int32)
    #throw output_img_sequences tensor
    # output_img_sequences = tf.cast(output_img_sequences, tf.int32)

    return input_img_sequences, input_ext_sequences, output_img_sequences


def load_data(filename):
    # create a queue
    # filename_queue = tf.train.string_input_producer([filename])

    # reader = tf.TFRecordReader()
    # return file_name and file
    # _, serialized_example = reader.read(filename_queue)
    dataset = tf.data.TFRecordDataset(filename)
    return dataset.map(parse_data)

def tec_root_mean_squared_error_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def tec_cosine_proximity_metric(y_true, y_pred):
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    # y_true = K.l2_normalize(y_true, axis=-1)
    # y_pred = K.l2_normalize(y_pred, axis=-1)
    # return K.sum(y_true * y_pred, axis=-1)
    return K.sum(K.l2_normalize(K.flatten(y_true), axis=-1) * K.l2_normalize(K.flatten(y_pred), axis=-1), axis=-1)
    
if __name__ == '__main__':
    cwd = os.getcwd()
    config = configparser.ConfigParser()
    config.read(os.path.join(cwd, 'dataset', "ion_dataset_info.ini"))

    img_rows              = Params.map_rows
    img_cols              = Params.map_cols
    input_time_steps      = Params.input_time_steps
    output_time_steps     = Params.output_time_steps
    external_dim          = Params.external_dim
    nb_train_samples      = config.getint('DatasetInfo', 'nb_train_samples')
    nb_validation_samples = config.getint('DatasetInfo', 'nb_validation_samples')
    nb_test_samples       = config.getint('DatasetInfo', 'nb_test_samples')
    nb_epoch              = 100
    batch_size            = Params.batch_size
    load_weights_path     = os.path.join(cwd, 'checkpoint', '20180820154635', "TEC_PRE_NET_MODEL_WEIGHTS.04-104.7666-0.86629.hdf5")
    save_weights_path     = os.path.join(cwd, 'checkpoint', datetime.now().strftime('%Y%m%d%H%M%S'))
    logs_path             = os.path.join(cwd, 'tensorboard', datetime.now().strftime('%Y%m%d%H%M%S'))    

    ion_dataset_normaliztion = MinMaxNormalization()

    ion_dataset          = load_data(os.path.join(cwd, 'dataset','ion_dataset.tfrecords'))
    ion_dataset_iterator = ion_dataset.make_initializable_iterator()
    ion_dataset_iterator_next_element = ion_dataset_iterator.get_next()

    training_dataset          = load_data(os.path.join(cwd, 'dataset','ion_training.tfrecords'))
    training_dataset_iterator = training_dataset.make_initializable_iterator()
    training_dataset_iterator_next_element = training_dataset_iterator.get_next()

    validation_dataset          = load_data(os.path.join(cwd, 'dataset','ion_validation.tfrecords'))
    validation_dataset_iterator = validation_dataset.make_initializable_iterator()
    validation_dataset_iterator_next_element = validation_dataset_iterator.get_next()

    input_img_sequences_training  = []
    input_ext_sequences_training  = []
    output_img_sequences_training = []

    input_img_sequences_validation  = []
    input_ext_sequences_validation  = []
    output_img_sequences_validation = []

    #开始一个会话
    # with tf.Session(config=tfconfig) as sess:
    with tf.Session() as sess:  
        sess.run(ion_dataset_iterator.initializer)
        sess.run(training_dataset_iterator.initializer)
        sess.run(validation_dataset_iterator.initializer)
       
        try:
            while True:
                input_img_sequences, input_ext_sequences, output_img_sequences = sess.run(ion_dataset_iterator_next_element)
                ion_dataset_normaliztion.fit(input_img_sequences)
                ion_dataset_normaliztion.fit(output_img_sequences)
        except tf.errors.OutOfRangeError:
            print("Dataset's MinMaxNormalization constructed. MinValue: {}, MaxValue: {}".format(ion_dataset_normaliztion._min, ion_dataset_normaliztion._max))

        try:
            while True:
                input_img_sequences, input_ext_sequences, output_img_sequences = sess.run(training_dataset_iterator_next_element)
                # input_img_sequences_training.append(ion_dataset_normaliztion.transform(input_img_sequences))
                # input_ext_sequences_training.append(input_ext_sequences)
                # output_img_sequences_training.append(ion_dataset_normaliztion.transform(output_img_sequences))
                input_img_sequences_training.append(input_img_sequences)
                input_ext_sequences_training.append(input_ext_sequences)
                output_img_sequences_training.append(output_img_sequences)
        except tf.errors.OutOfRangeError:
            print("Training dataset constructed ...")        

        try:
            while True:
                input_img_sequences, input_ext_sequences, output_img_sequences = sess.run(validation_dataset_iterator_next_element)
                # input_img_sequences_validation.append(ion_dataset_normaliztion.transform(input_img_sequences))
                # input_ext_sequences_validation.append(input_ext_sequences)
                # output_img_sequences_validation.append(ion_dataset_normaliztion.transform(output_img_sequences))
                input_img_sequences_validation.append(input_img_sequences)
                input_ext_sequences_validation.append(input_ext_sequences)
                output_img_sequences_validation.append(output_img_sequences)
        except tf.errors.OutOfRangeError:
            print("Validation dataset constructed ...")   


    input_img_sequences_training_fit_x = np.array(input_img_sequences_training)
    input_ext_sequences_training_fit_x = np.array(input_ext_sequences_training)
    output_img_sequences_training_fit_y = np.array(output_img_sequences_training)

    input_img_sequences_validation_fit_x = np.array(input_img_sequences_validation)
    input_ext_sequences_validation_fit_x = np.array(input_ext_sequences_validation)
    output_img_sequences_validation_fit_y = np.array(output_img_sequences_validation)



    def generator(model_input_1, model_input_2, model_output_1, batch_size):
        sample_size = model_input_1.shape[0]
        indexs = np.arange(sample_size)
        np.random.shuffle(indexs)
        batches = [indexs[range(batch_size*i, min(sample_size, batch_size*(i+1)))] for i in range(sample_size//batch_size+1)]
        while True:
            for i in batches:
                yield [model_input_1[i], model_input_2[i]], model_output_1[i]


    model = tec_pre_net((img_rows, img_cols), input_time_steps, output_time_steps, external_dim)
    model.load_weights(load_weights_path, by_name=False)

    opt = Adam(lr=Params.lr, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss=tec_root_mean_squared_error_loss, metrics=[tec_cosine_proximity_metric])

    # Callback
    try:
        os.makedirs(save_weights_path)
    except:
        pass
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(save_weights_path, 'TEC_PRE_NET_MODEL_WEIGHTS.{epoch:02d}-{val_loss:.4f}-{val_tec_cosine_proximity_metric:.5f}.hdf5'),
        monitor='val_acc',
        verbose=1,
        save_weights_only= True,
        save_best_only=False
    )
    
    try:
        os.makedirs(logs_path)
    except:
        pass
    tensorboarder = TensorBoard(
        log_dir=logs_path,
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None
    )

    model.fit_generator(
        generator(input_img_sequences_training_fit_x, input_ext_sequences_training_fit_x, output_img_sequences_training_fit_y, batch_size),
        epochs=nb_epoch,
        verbose=1,
        callbacks=[checkpointer, tensorboarder],
        validation_data=generator(input_img_sequences_validation_fit_x, input_ext_sequences_validation_fit_x, output_img_sequences_validation_fit_y, batch_size),
        steps_per_epoch=nb_train_samples//batch_size,
        validation_steps=nb_validation_samples//batch_size
    )
