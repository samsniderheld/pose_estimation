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

    X = np.empty((1, 5,2))
    Y = np.empty((1, 52,3))

    X[0] = np.load(input_paths[rand_idx])
    Y[0] = np.load(output_paths[rand_idx])

    # with open(input_paths[rand_idx]) as file:
    #             csv_reader = csv.reader(file, delimiter=',')
    #             for j,row in enumerate(csv_reader):
    #                 for k, val in enumerate(row[1:3]):
    #                   X[0,j,k] = float(val)/1024

    # with open(output_paths[rand_idx]) as file:
    #             csv_reader = csv.reader(file, delimiter=',')
    #             for j,row in enumerate(csv_reader):
    #                 for k, val in enumerate(row[3:6]):
    #                   Y[0,j,k] = float(val)/1024



    return X, Y




