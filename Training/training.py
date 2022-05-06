import os
from Model.pose_detection_model import create_pose_detector
from Model.img_2_bone import create_img_2_bone
from Utils.reporting import *
from Data_Utils.generate_samples import *
from Data_Utils.data_generator import DataGenerator, ImageDataGenerator


def train(args):

    #setup data
    random_sample = get_random_sample(args)

    data_generator = DataGenerator(args,shuffle=True)

    pose_detector = create_pose_detector()

    all_history = []
    lowest_loss = 10000

    output_test_csv_dir = os.path.join(args.base_results_dir,args.output_test_csv_dir)
    model_save_path = os.path.join(args.base_results_dir, args.saved_model_dir)
    output_history_path = os.path.join(args.base_results_dir,args.history_dir)

    for i,epoch in enumerate(range(args.num_epochs)):

        print(f'training epoch {str(i)}')

        history = pose_detector.fit(
            data_generator,
            epochs=1,
            shuffle=True,
        )

        loss = round(history.history['loss'][0],8)

        if(args.save_best_only):

            if(loss < lowest_loss):
                generate_bone_accuracy_table(pose_detector,random_sample, 
                    output_test_csv_dir, i, args.print_csv)



                pose_detector_save_path = os.path.join(model_save_path,"pose_detector_model.h5")
                pose_detector.save_weights(pose_detector_save_path)

                lowest_loss = loss

        else:
            pose_detector_save_path = os.path.join(model_save_path,f"{i:04d}_pose_detector_model.h5")
            pose_detector.save_weights(pose_detector_save_path)

        all_history.append(history.history['loss'][0])
        save_experiment_history(args,all_history,output_history_path)

    generate_bone_accuracy_table(pose_detector,random_sample,
                output_history_path, args.num_epochs, args.print_csv)

    save_experiment_history(args,all_history,output_history_path)

    if(not args.save_best_only):
        pose_detector_save_path = os.path.join(model_save_path,f"{args.num_epochs:04d}_pose_detector_model.h5")
        pose_detector.save_weights(pose_detector_save_path)


def train_img_2_bone(args):

    #setup data
    random_sample = get_random_img_sample(args)

    base_bone_path = os.path.join(args.base_data_dir, "bone_connections.csv")

    base_bones = load_base_bones(base_bone_path)

    data_generator = ImageDataGenerator(args,shuffle=True)

    pose_detector = create_img_2_bone()

    all_history = []
    lowest_loss = 10000

    output_test_csv_dir = os.path.join(args.base_results_dir,args.output_test_csv_dir)
    model_save_path = os.path.join(args.base_results_dir, args.saved_model_dir)
    output_history_path = os.path.join(args.base_results_dir,args.history_dir)

    for i,epoch in enumerate(range(args.num_epochs)):

        print(f'training epoch {str(i)}')

        history = pose_detector.fit(
            data_generator,
            epochs=1,
            shuffle=True,
        )

        loss = round(history.history['loss'][0],8)

        if(args.save_best_only):

            if(loss < lowest_loss):
                generate_bone_accuracy_table(pose_detector,random_sample, 
                    output_test_csv_dir, i, args.print_csv)

                plot_skeletons(pose_detector,base_bones, random_sample, 
                    output_test_csv_dir, i)

                pose_detector_save_path = os.path.join(model_save_path,"pose_detector_model.h5")
                pose_detector.save_weights(pose_detector_save_path)

                lowest_loss = loss

        else:
            pose_detector_save_path = os.path.join(model_save_path,f"{i:04d}_pose_detector_model.h5")
            pose_detector.save_weights(pose_detector_save_path)

        all_history.append(history.history['loss'][0])
        save_experiment_history(args,all_history,output_history_path)

    generate_bone_accuracy_table(pose_detector,random_sample,
                output_history_path, args.num_epochs, args.print_csv)

    save_experiment_history(args,all_history,output_history_path)

    if(not args.save_best_only):
        pose_detector_save_path = os.path.join(model_save_path,f"{args.num_epochs:04d}_pose_detector_model.h5")
        pose_detector.save_weights(pose_detector_save_path)

