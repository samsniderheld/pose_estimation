import tensorflow as tf
import numpy as np
import glob
import random
import os
import cv2
import csv


class DataGenerator(tf.keras.utils.Sequence):
#     'Generates data for tf.keras'
    def __init__(self, args, shuffle=True,):
        self.shuffle = shuffle
        # self.input_dir = os.path.join(args.base_data_dir,args.input_data_dir + "*")
        # self.output_dir = os.path.join(args.base_data_dir,args.output_data_dir + "*")
        self.input_dir = "Normalized_Data/x_data/*"
        self.output_dir = "Normalized_Data/y_data/*"
        self.batch_size = args.batch_size
        self.input_files  = sorted(glob.glob(self.input_dir))
        self.output_dir  = sorted(glob.glob(self.output_dir))
        self.all_files = list(zip(self.input_files,self.output_dir))
        self.on_epoch_end()
        self.count = self.__len__()
        print("number of all samples = ", len(self.input_files))


    def __len__(self):
        'Denotes the number of batches per epoch'
        self.num_batches = int(np.floor(len(self.all_files) / self.batch_size))
        return self.num_batches

    def __getitem__(self, index):
      
        X,Y = self.__data_generation(index)

        return X,Y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.all_files)


    def __data_generation(self, idx):
        'Generates data containing batch_size samples' 

        batch_files = self.all_files[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        
        X = np.empty((self.batch_size,4,2))
        Y = np.empty((self.batch_size,52,3))

        # read image
        for i, batch_file in enumerate(batch_files):

            X[i] = np.load(batch_file[0])
            Y[i] = np.load(batch_file[1])
            # with open(batch_file[0]) as file:
            #     csv_reader = csv.reader(file, delimiter=',')
            #     for j,row in enumerate(csv_reader):
            #         for k, val in enumerate(row[1:3]):
            #           X[i,j,k] = float(val)/1024

            # with open(batch_file[1]) as file:
            #     csv_reader = csv.reader(file, delimiter=',')
            #     for j,row in enumerate(csv_reader):
            #         for k, val in enumerate(row[3:6]):
            #           Y[i,j,k] = float(val)/1024
            #         # for k, val in enumerate(row[4:7]):
            #         #   Y[i,j,k] = float(val)/360
            
        return X,Y

class ImageDataGenerator(tf.keras.utils.Sequence):
#     'Generates data for tf.keras'
    def __init__(self, args, shuffle=True,):
        self.shuffle = shuffle
        # self.input_dir = os.path.join(args.base_data_dir,args.input_data_dir + "*")
        # self.output_dir = os.path.join(args.base_data_dir,args.output_data_dir + "*")
        self.input_dir = "Normalized_Data/x_data/*"
        self.output_dir = "Normalized_Data/y_data/*"
        self.batch_size = args.batch_size
        self.input_files  = sorted(glob.glob(self.input_dir))
        self.output_dir  = sorted(glob.glob(self.output_dir))
        self.all_files = list(zip(self.input_files,self.output_dir))
        self.weight_matrix = tf.linspace([1.0,1.0,1.0],[.1,.1,.1],52)
        self.on_epoch_end()
        self.count = self.__len__()
        print("number of all samples = ", len(self.input_files))


    def __len__(self):
        'Denotes the number of batches per epoch'
        self.num_batches = int(np.floor(len(self.all_files) / self.batch_size))
        return self.num_batches

    def __getitem__(self, index):
      
        X,Y = self.__data_generation(index)

        return X,Y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.all_files)


    def __data_generation(self, idx):
        'Generates data containing batch_size samples' 

        batch_files = self.all_files[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        
        X = np.empty((self.batch_size,128,128,1))
        Y = np.empty((self.batch_size,52,3))
        Y_Weight = np.empty((self.batch_size,52,3))
        # read image
        for i, batch_file in enumerate(batch_files):

            X[i] = np.load(batch_file[0])
            Y[i] = np.load(batch_file[1])
            Y_Weight[i] = self.weight_matrix

            
        # return X,Y
        return ([X, Y, Y_Weight], Y)

