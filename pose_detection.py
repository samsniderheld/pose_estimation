import argparse
import os
import shutil
from Training.training import train
from Data_Utils.normalize import normalize_data
from datetime import datetime


def parse_args():
    desc = "An autoencoder for pose similarity detection"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--base_data_dir', type=str, default="Data/", help='The directory that holds the image data')
    parser.add_argument('--input_data_dir', type=str, default="CSV/", help='The directory for CSV input data')
    parser.add_argument('--output_data_dir', type=str, default="CSV/", help='The directory for CSV input data')
    parser.add_argument('--base_results_dir', type=str, default="/", help='The base directory to hold the results')
    parser.add_argument('--output_test_csv_dir', type=str, default="CSV/", help='The directory for result csvs')
    parser.add_argument('--saved_model_dir', type=str, default="Saved_Models/", help='The directory for input data')
    parser.add_argument('--history_dir', type=str, default="History/", help='The directory for input data')
    parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--csv_dims', type=int, default=156, help='The number of csv channels')
    parser.add_argument('--input_channels', type=int, default=2, help='The number of input bone dims')
    parser.add_argument('--output_channels', type=int, default=2, help='The number of input bone dims')
    parser.add_argument('--print_freq', type=int, default=5, help='How often is the status printed')
    parser.add_argument('--save_freq', type=int, default=10, help='How often is the model saved')
    parser.add_argument('--save_best_only', action='store_true')
    parser.add_argument('--print_csv', action='store_true')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--notes', type=str, default="N/A", help='A description of the experiment')
    parser.add_argument('--experiment_name', type=str, default="", help='A name for the experiment')

    return parser.parse_args()


def main():
    args = parse_args()

    args.experiment_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + args.experiment_name

    args.base_results_dir = os.path.join(args.base_results_dir,args.experiment_name)

    if(not os.path.exists(args.base_results_dir)):
        os.makedirs(args.base_results_dir)
        os.makedirs(os.path.join(args.base_results_dir,"CSV"))
        os.makedirs(os.path.join(args.base_results_dir,"History"))
        os.makedirs(os.path.join(args.base_results_dir,"Saved_Models"))


    normalize_data(args)

    train(args)
        

    print("done training")


if __name__ == '__main__':
    main()
