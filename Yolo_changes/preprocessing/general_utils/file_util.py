import os
import csv
import random
import numpy as np

from preprocessing.general_utils import bb_util, img_util


def read_annot(file_path):
    '''
    This function reads the annotation file
    file_path : path to the annotation file
    return : list of obj class and annotations
    '''
    annot_list = []
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            row = [float(x) for x in row]
            annot_list.append((row[0], row[1:]))
    f.close()
    return annot_list


def write_annot(obj_bb_list, file_path):
    '''
    This function writes the annotation file
    obj_bb_list : list of obj class and annotations
    file_path : path to save the annotation file
    '''
    obj_list, bb_list = obj_bb_list
    with open(file_path, 'w') as f:
        for obj_id, bb in zip(obj_list, bb_list):
            annot = [int(obj_id)] + bb
            csv.writer(f, delimiter=' ').writerow(annot)


def make_list(dir_path):
    '''
    This function makes a list of images and the corresponding annotation files path
    dir_path : path to the directory containing images(*.png) and annotations(*.txt)
    return : list of images and the corresponding annotation files path
    '''
    try:
        list_annot = []
        list_img = []
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                if name.endswith(".txt"):
                    annot_file_path = os.path.join(root, name)
                    img_file_path = annot_file_path.replace("annot_", "").replace(".txt", ".png")
                    if os.path.exists(img_file_path) and os.path.exists(annot_file_path):
                        list_img.append(img_file_path)
                        list_annot.append(annot_file_path)

        if len(list_img) != len(list_annot):
            print("Error : Length of annotation and images list must be equal!!!")
            return []
        else:
            final_list = [(img, annot) for img, annot in zip(list_img, list_annot)]
            return final_list
    except:
        print("Error : The path {} doesn't exist!!!".format(dir_path))


def generate_yolo_dataset_single_obj(dir_path_input, dir_path_output, padding=20, desired_size=[500, 500],
                                     train_val_test=[0.7, 0.2, 0.1], seed=0):
    '''
    This function generates the dataset for each single object(tool) separately
    in a way that the output image contains only one single object
    dir_path_input : path to the directory containing images(*.png) and annotations(*.txt)
    dir_path_output : path to the directory to save the output images and annotations (final dataset)
    padding : padding size to the final cropped images (pixels)
    desired_size : maximum bounding box size (pixels)
    train_val_test : Train, Validation and Test proportion
    seed : Global random seed
    '''

    if not os.path.exists(dir_path_output):
        os.makedirs(dir_path_output, exist_ok=True)

    all_file_list = make_list(dir_path_input)
    random.seed(seed)
    random.shuffle(all_file_list)
    n = len(all_file_list)

    indx = [int(p * n) for p in train_val_test]
    indx_list = [[0, indx[0]], [indx[0], indx[0] + indx[1]], [indx[0] + indx[1], n]]
    mode_dict = {"train": indx_list[0], "valid": indx_list[1], "test": indx_list[2]}

    for mode, indx_range in mode_dict.items():
        current_output = os.path.join(dir_path_output, mode)
        os.makedirs(current_output, exist_ok=True)
        data_list = all_file_list[indx_range[0]:indx_range[1]]
        os.makedirs(os.path.join(current_output, "images"), exist_ok=True)
        os.makedirs(os.path.join(current_output, "labels"), exist_ok=True)
        for item in data_list:
            input_img_file, input_annot_file = item[0], item[1]
            # open image and annot files
            print("processing and generating from : ", input_img_file)
            img = img_util.read_image(input_img_file)
            (img_height, img_width, _) = np.shape(img)
            annot_list = read_annot(input_annot_file)
            obj_list = [bb[0] for bb in annot_list]
            bb_list = [bb[1] for bb in annot_list]

            # convert yolo annot to abs annot
            bb_list = bb_util.convert_yolo_bb_to_abs(bb_list, [img_width, img_height])
            # find the largest bbox contour including all bboxes
            bbox_l = bb_util.find_largest_contour(bb_list)
            # drop the sample if the bounding box is larger than the desired size
            if bbox_l[2] > desired_size[0] or bbox_l[3] > desired_size[1]:
                continue

            # for each object in the image crop and generate 5 new images
            # where the bounding box for 4 are fitted to the edges of the image (top,left,right,bottom) and 1 centered
            for i in range(len(obj_list)):
                for m in range(5):
                    # random crop the image and get new annot
                    new_img, new_annot_list = img_util.rnd_crop_image_bb_side_fit(img, bb_list[i], [bb_list[i]],
                                                                                  pad_size=padding, max_try=10, mode=m)
                    if new_img is None:
                        pass
                    else:
                        new_annot_list = bb_util.convert_abs_bb_to_yolo(new_annot_list, img_shape=[np.shape(new_img)[1],
                                                                                                   np.shape(new_img)[
                                                                                                       0]])
                        # save the img and the new annot file
                        new_name = input_img_file.split("/")[-3] + "_" + input_img_file.split("/")[-1]
                        new_name = new_name.replace(".png", "_" + str(m) + "_" + str(i) + ".png")
                        output_img_file = os.path.join(current_output, "images") + "/" + new_name
                        output_annot_file = os.path.join(current_output, "labels") + "/" + new_name.replace(".png",
                                                                                                            ".txt")
                        img_util.write_image(new_img, output_img_file)
                        write_annot([[obj_list[i]], new_annot_list], output_annot_file)


def generate_yolo_dataset_multi_obj(dir_path_input, dir_path_output, padding=20, desired_size=[500, 500],
                                    train_val_test=[0.7, 0.2, 0.1], seed=0):
    '''
    This function generates the dataset for pair of objects(tools) separately
    by finding the largest bounding box between two pairs in the original multi object image
    dir_path_input : path to the directory containing images(*.png) and annotations(*.txt)
    dir_path_output : path to the directory to save the output images and annotations (final dataset)
    padding : padding size to the final cropped images (pixels)
    desired_size : maximum bounding box size (pixels)
    train_val_test : Train, Validation and Test proportion
    seed : Global random seed
    '''

    if not os.path.exists(dir_path_output):
        os.makedirs(dir_path_output, exist_ok=True)

    all_file_list = make_list(dir_path_input)
    random.seed(seed)
    random.shuffle(all_file_list)
    n = len(all_file_list)

    indx = [int(p * n) for p in train_val_test]
    indx_list = [[0, indx[0]], [indx[0], indx[0] + indx[1]], [indx[0] + indx[1], n]]
    mode_dict = {"train": indx_list[0], "valid": indx_list[1], "test": indx_list[2]}

    for mode, indx_range in mode_dict.items():
        current_output = os.path.join(dir_path_output, mode)
        os.makedirs(current_output, exist_ok=True)
        data_list = all_file_list[indx_range[0]:indx_range[1]]
        os.makedirs(os.path.join(current_output, "images"), exist_ok=True)
        os.makedirs(os.path.join(current_output, "labels"), exist_ok=True)
        for item in data_list:
            input_img_file, input_annot_file = item[0], item[1]
            # open image and annot files
            print("processing and generating from : ", input_img_file)
            img = img_util.read_image(input_img_file)
            (img_height, img_width, _) = np.shape(img)
            annot_list = read_annot(input_annot_file)
            obj_list = [bb[0] for bb in annot_list]
            bb_list = [bb[1] for bb in annot_list]

            # convert yolo annot to abs annot
            bb_list = bb_util.convert_yolo_bb_to_abs(bb_list, [img_width, img_height])
            # for each of the objects in the image find a pairing object and crop the image
            for i in range(len(obj_list)):
                candid_obj_list = obj_list[i + 1:]
                candid_bb_list = bb_list[i + 1:]
                for j in range(len(candid_obj_list)):
                    # find the largest bbox contour including all bboxes
                    bbox_l = bb_util.find_largest_contour([bb_list[i], candid_bb_list[j]])
                    # drop the sample if the bounding box is larger than the desired size
                    if bbox_l[2] > desired_size[0] or bbox_l[3] > desired_size[1]:
                        continue
                    else:
                        final_obj_bb = [(obj_list[z], bb_list[z]) for z in range(len(obj_list)) if
                                        bb_util.bb_intersection(bbox_l[:], bb_list[z][:])]
                        final_obj_list = [obj_bb[0] for obj_bb in final_obj_bb]
                        final_bb_list = [obj_bb[1] for obj_bb in final_obj_bb]
                        # for each pair of objects in the image crop and generate 5 new images
                        # where the largest contour bounding box for 4 are fitted to the edges of the image (top,left,right,bottom) and 1 centered
                        for m in range(5):
                            # random crop the image and get new annot
                            new_img, new_annot_list = img_util.rnd_crop_image_bb_side_fit(img, bbox_l, final_bb_list,
                                                                                          pad_size=padding, max_try=10,
                                                                                          mode=m)

                            if new_img is None:
                                pass
                            else:
                                # convert back the abs annot to yolo annot
                                new_annot_list = bb_util.convert_abs_bb_to_yolo(new_annot_list,
                                                                                img_shape=[np.shape(new_img)[1],
                                                                                           np.shape(new_img)[0]])
                                # save the img and the new annot file
                                new_name = input_img_file.split("/")[-2] + "_" + input_img_file.split("/")[-1]
                                new_name = new_name.replace(".png", "_" + str(i) + "_" + str(j) + "_" + str(m) + ".png")
                                output_img_file = os.path.join(current_output, "images") + "/" + new_name
                                output_annot_file = os.path.join(current_output, "labels") + "/" + new_name.replace(
                                    ".png", ".txt")
                                img_util.write_image(new_img, output_img_file)
                                write_annot([final_obj_list, new_annot_list], output_annot_file)
