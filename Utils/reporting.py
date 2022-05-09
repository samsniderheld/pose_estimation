import cv2
import numpy as np
import os
import csv
import pandas as pd
import json
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt
from Data_Utils.generate_samples import *

def generate_bone_accuracy_table(model, test_input, path, epoch, print_to_terminal):

  prediction = model(test_input[0])


  outputs = prediction.numpy().flatten()

  dict = { 
    'Input Values': test_input[1].flatten(),
    'Predictions': outputs
  }

  df = pd.DataFrame(dict)

  output_csv_path = os.path.join(path,f"epoch_{epoch:04d}_bones.csv")

  df.to_csv(output_csv_path)  

  if print_to_terminal:
    print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

def save_experiment_history(args, history, path):

  experiment = {

      'notes': args.notes,
      'number_of_epochs': args.num_epochs,
      'batch_size': args.batch_size,
      'loss_history': history

  }

  file_name = args.experiment_name + ".json"

  output_path = os.path.join(path,file_name)

  with open(output_path, 'w') as outfile:
    json.dump(experiment, outfile)


def load_base_bones(path):
  base_bones = np.empty((52,2),dtype=np.int32)

  with open(path) as file:
      csv_reader = csv.reader(file, delimiter=',')
      for j,row in enumerate(csv_reader):
        for l, connection in enumerate(row[0:2]):
          base_bones[j,l] = float(connection)

  return base_bones


def plot_skeletons(model, base_bones, path, epoch, num):

  for i in range(0,num):

    X,Y = get_random_img_sample()

    output = model(X)

    prediction = output.numpy()



    fig = plt.figure()

    ax_img = fig.add_subplot(1,3,1)
    plt.imshow(X[0])

    
    ax_g = fig.add_subplot(1, 3, 2, projection='3d')
    for connections in enumerate(base_bones):
      bone = [Y[0][idx] for idx in connections[1]]
      bone = np.array(bone)
      x,y,z = np.split(bone.T,3)
      ax_g.plot(x[0],z[0],y[0])

    ax_p = fig.add_subplot(1, 3, 3, projection='3d')
    for connections in enumerate(base_bones):
      bone = [prediction[0][idx] for idx in connections[1]]
      bone = np.array(bone)
      x,y,z = np.split(bone.T,3)
      ax_p.plot(x[0],z[0],y[0])

    output_path = os.path.join(path,f"epoch_{epoch:04d}_skeleton_{i:04d}.jpg")

    plt.savefig(output_path)
    plt.close()






