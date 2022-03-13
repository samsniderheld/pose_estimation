import tensorflow as tf
import numpy as np
import random
import glob
import os
import cv2
import csv
from Utils.util_functions import *


def get_random_sample(args):

    input_data_path = os.path.join(args.base_data_dir,args.input_data_dir + "*")

    input_paths = sorted(glob.glob(input_data_path), key=natural_keys)

    rand_idx = random.randint(0,len(input_paths))

    output_data_path = os.path.join(args.base_data_dir,args.input_data_dir + "*")

    output_paths = sorted(glob.glob(output_data_path),key=natural_keys)

    X = np.empty((1, 52,2))
    Y = np.empty((1, 52,3))

    with open(input_paths[rand_idx]) as file:
                csv_reader = csv.reader(file, delimiter=',')
                for j,row in enumerate(csv_reader):
                    for k, val in enumerate(row[4:6]):
                      X[0,j,k] = float(val)/360

    with open(output_paths[rand_idx]) as file:
                csv_reader = csv.reader(file, delimiter=',')
                for j,row in enumerate(csv_reader):
                    for k, val in enumerate(row[4:7]):
                      Y[0,j,k] = float(val)/360



    return X, Y




