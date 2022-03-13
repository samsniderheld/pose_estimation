import cv2
import numpy as np
import os
import csv
import pandas as pd
import json
from datetime import datetime
from tabulate import tabulate

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







