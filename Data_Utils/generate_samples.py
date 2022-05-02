import tensorflow as tf
import numpy as np
import random
import glob
import os
import cv2
import csv
from Utils.util_functions import *


def get_random_sample(args):

    input_data_path = "Normalized_Data/x_data/*"

    input_paths = sorted(glob.glob(input_data_path), key=natural_keys)

    rand_idx = random.randint(0,len(input_paths))

    output_data_path = "Normalized_Data/y_data/*"

    output_paths = sorted(glob.glob(output_data_path),key=natural_keys)

    X = np.empty((1, 4,2))
    Y = np.empty((1, 52,3))

    X[0] = np.load(input_paths[rand_idx])
    Y[0] = np.load(output_paths[rand_idx])



    return X, Y


def get_random_img_sample(args):

    input_data_path = "Normalized_Data/x_data/*"

    input_paths = sorted(glob.glob(input_data_path), key=natural_keys)

    rand_idx = random.randint(0,len(input_paths))

    output_data_path = "Normalized_Data/y_data/*"

    output_paths = sorted(glob.glob(output_data_path),key=natural_keys)

    X = np.empty((1, 128,128,3))
    Y = np.empty((1, 52,3))
    Y_Weight = np.empty((1,52,3))

    X[0] = np.load(input_paths[rand_idx])
    Y[0] = np.load(output_paths[rand_idx])
    Y_Weight[0] = tf.linspace([1.0,1.0,1.0],[.1,.1,.1],52)





    return ([X,Y,Y_Weight], Y)



