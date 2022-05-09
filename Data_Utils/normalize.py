from numpy.linalg import norm
import tensorflow as tf
import numpy as np
import random
import glob
import os
import cv2
import csv
from Utils.util_functions import *
from tqdm import tqdm


def normalize_data(args):

    if(not os.path.exists("Normalized_Data")):
      os.makedirs("Normalized_Data")
      os.makedirs("Normalized_Data/x_data")
      os.makedirs("Normalized_Data/y_data")

    output_path = "Normalized_Data"
    output_path_x = os.path.join(output_path,"x_data")
    output_path_y = os.path.join(output_path,"y_data")



    input_data_path = os.path.join(args.base_data_dir,args.input_data_dir + "*")
    input_paths = sorted(glob.glob(input_data_path), key=natural_keys)

    output_data_path = os.path.join(args.base_data_dir,args.output_data_dir + "*")
    output_paths = sorted(glob.glob(output_data_path), key=natural_keys)

    all_data_x = []
    all_data_y = []

    print("loading data")
    for i, path in tqdm(enumerate(input_paths)):

        X = np.empty((4,2),dtype=np.float64)

        with open(path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            for j,row in enumerate(csv_reader):
                for k, val_x in enumerate(row[1:3]):
                  X[j,k] = float(val_x)

        all_data_x.append(X)

    for i, path in tqdm(enumerate(output_paths)):

        Y = np.empty((52,3),dtype=np.float64)

        with open(path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            for j,row in enumerate(csv_reader):
                for l, val_y in enumerate(row[4:-1]):
                  Y[j,l] = float(val_y)

        all_data_y.append(Y)


    print("calculating mean and std")
    np_all_data_x = np.array(all_data_x,dtype=np.float64)
    np_all_data_y = np.array(all_data_y,dtype=np.float64)

    print(np_all_data_x.shape)
    print(np_all_data_y.shape)



    data_y_mean = np.mean(np_all_data_y,dtype=np.float64)
    data_y_std = np.std(np_all_data_y,dtype=np.float64)


    print("normalizing and saving data")
    for i in tqdm(range(0,np_all_data_x.shape[0])):

        normalized_x_sample = np_all_data_x[i]/1024
        normalized_y_sample = np_all_data_y[i]/360


        np.save(os.path.join(output_path_x,f"{i:04d}_x_data"), normalized_x_sample)
        np.save(os.path.join(output_path_y,f"{i:04d}_y_data"), normalized_y_sample)

    np.savetxt("x.csv", normalized_x_sample, delimiter=",")
    np.savetxt("y.csv", normalized_y_sample, delimiter=",")


def normalize_image_data(args):

    if(not os.path.exists("Normalized_Data")):
      os.makedirs("Normalized_Data")
      os.makedirs("Normalized_Data/x_data")
      os.makedirs("Normalized_Data/y_data")

    output_path = "Normalized_Data"
    output_path_x = os.path.join(output_path,"x_data")
    output_path_y = os.path.join(output_path,"y_data")



    input_data_path = os.path.join(args.base_data_dir,args.input_img_data_dir + "*")
    input_paths = sorted(glob.glob(input_data_path), key=natural_keys)

    output_data_path = os.path.join(args.base_data_dir,args.output_data_dir + "*")
    output_paths = sorted(glob.glob(output_data_path), key=natural_keys)

    all_data_x = []
    all_data_y = []

    blur_size = (5, 5)

    print("loading data")
    for i, path in tqdm(enumerate(input_paths)):

        X = cv2.imread(path)

        X = cv2.blur(X, blur_size)

        all_data_x.append(X)

    for i, path in tqdm(enumerate(output_paths)):

        Y = np.empty((52,3),dtype=np.float64)

        with open(path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            for j,row in enumerate(csv_reader):
                for l, val_y in enumerate(row[1:4]):
                  Y[j,l] = float(val_y)

        all_data_y.append(Y)


    np_all_data_x = np.array(all_data_x,dtype=np.float64)
    np_all_data_y = np.array(all_data_y,dtype=np.float64)

    print(np_all_data_x.shape)
    print(np_all_data_y.shape)


    print("normalizing and saving data")
    for i in tqdm(range(0,np_all_data_x.shape[0])):

        normalized_x_sample = np_all_data_x[i]/255
        normalized_y_sample = np_all_data_y[i]


        np.save(os.path.join(output_path_x,f"{i:04d}_x_data"), normalized_x_sample)
        np.save(os.path.join(output_path_y,f"{i:04d}_y_data"), normalized_y_sample)







