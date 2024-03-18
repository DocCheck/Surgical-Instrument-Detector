import argparse
import os
from preprocessing.general_utils import file_util


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-single", "--input-data-path", type=str, default="data/dataset/Orig_OPBesteck_dataset_single/",
                        help="Raw dataset path folder for single objects")
    parser.add_argument("--input-data-multi", type=str, default="data/dataset/Orig_OPBesteck_dataset_multi/",
                        help="Raw dataset path folder for multi objects")
    parser.add_argument("--output-data-path", type=str, default="data/dataset/Rona_dataset_final/",
                        help="Preprocessed dataset path folder")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--bb-size", type=int, default=640, help="maximum bounding box size (pixels)")
    parser.add_argument("--padding", type=int, default=50, help="Padding size to the final cropped images (pixels)")
    parser.add_argument("--train-val-test", type=float, nargs='+', default=[0.9, 0.1, 0.0],
                        help="Train, Validation and Test proportion", required=True)
    return parser.parse_known_args()[0] if known else parser.parse_args()


def generate_dataset(opt):
    if os.path.exists(opt.input_data_single):
        file_util.generate_yolo_dataset_single_obj(dir_path_input=opt.input_data_single,
                                                   dir_path_output=opt.output_data_path, padding=opt.padding,
                                                   desired_size=[opt.bb_size, opt.bb_size],
                                                   train_val_test=opt.train_val_test, seed=opt.seed)
        if os.path.exists(opt.input_data_multi):
            file_util.generate_yolo_dataset_multi_obj(dir_path_input=opt.input_data_multi,
                                                      dir_path_output=opt.output_data_path,
                                                      padding=opt.padding, desired_size=[opt.bb_size, opt.bb_size],
                                                      train_val_test=opt.train_val_test, seed=opt.seed)
        else:
            print("The dataset path for multi objects does not exist !!!")
    else:
        print("The dataset path does not exist !!!")


if __name__ == "__main__":
    opt = parse_opt()
    generate_dataset(opt)
