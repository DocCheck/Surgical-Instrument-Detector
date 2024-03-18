import os
import csv
import cv2
import random
import numpy as np

import bb_util
import img_util


def read_image(img_file):
    img = cv2.imread(img_file)
    return img


def read_annot(file_path):
    annot_list = []
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            row = [float(x) for x in row]
            annot_list.append((row[0], row[1:]))
    f.close()
    return annot_list


def write_image(img, img_file):
    cv2.imwrite(img_file, img)


def write_annot(obj_bb_list, file_path):
    obj_list, bb_list = obj_bb_list
    #print(obj_list)
    #print(bb_list)
    with open(file_path, 'w') as f:
        for obj_id, bb in zip(obj_list, bb_list):
            annot = [int(obj_id)] + bb
            csv.writer(f, delimiter=' ').writerow(annot)



def make_list(dir_path):
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


def create_yolo_dataset(dir_path_input, dir_path_output, seed, desired_size):
    '''
    This function generates the dataset for all objects(tools)
    by finding the largest bounding box in the original image
    '''

    if not os.path.exists(dir_path_output):
        os.makedirs(dir_path_output, exist_ok=True)

    all_file_list = make_list(dir_path_input)
    print(all_file_list)
    random.seed(seed)
    random.shuffle(all_file_list)
    #all_file_list = all_file_list[:10]
    train_test_valid_prob = [0.8, 0.2, 0.0]
    n = len(all_file_list)

    indx = [int(p*n) for p in train_test_valid_prob]
    indx_list = [[0,indx[0]],[indx[0],indx[0]+indx[1]],[indx[0]+indx[1],n]]
    mode_dict = {"train": indx_list[0], "valid": indx_list[1], "test": indx_list[2]}
    print(indx_list)

    for mode, indx_range in mode_dict.items():
        current_output = os.path.join(dir_path_output, mode)
        os.makedirs(current_output, exist_ok=True)
        data_list = all_file_list[indx_range[0]:indx_range[1]]
        os.makedirs(os.path.join(current_output,"images"), exist_ok=True)
        os.makedirs(os.path.join(current_output,"labels"), exist_ok=True)
        for item in data_list:
            input_img_file, input_annot_file = item[0], item[1]
            # open image and annot files
            img = read_image(input_img_file)
            (img_height, img_width, _) = np.shape(img)
            annot_list = read_annot(input_annot_file)
            print(annot_list)
            obj_list = [bb[0] for bb in annot_list]
            bb_list = [bb[1] for bb in annot_list]
            print(obj_list)
            #print("***********************")
            #print(obj_id, annot)
            # convert yolo annot to abs annot
            bb_list = bb_util.convert_yolo_bb_to_abs(bb_list, [img_width, img_height])
            print(bb_list)
            # find the largest bbox contour including all bboxes
            bbox_l = bb_util.find_largest_contour(bb_list)
            print(bbox_l)
            if bbox_l[2]>desired_size[0] or bbox_l[3]>desired_size[1]:
                continue

            # crop image and get new annot
            #print(obj_id, annot)
            #new_img, new_annot_list = img_util.rnd_crop_img(img, bbox_l, bb_list, desired_size=desired_size)
            for m in range(5):
                #m=4
                new_img, new_annot_list = img_util.rnd_crop_img_bb_side_fit(img, bbox_l, bb_list, desired_size=desired_size, stride=50, mode=m)
                #new_img, new_annot_list = img_util.rnd_crop_img(img, bbox_l, bb_list, desired_size=desired_size)

                print(new_annot_list)
                print(desired_size)
                # convert back the abs annot to yolo annot
                new_annot_list = bb_util.convert_abs_bb_to_yolo(new_annot_list, img_shape=[np.shape(new_img)[1], np.shape(new_img)[0]])
                #test_annot = bb_util.convert_yolo_bb_to_abs(new_annot,desired_size)
                print(obj_list, new_annot_list)

                #print(obj_id, test_annot)
                #bb_util.draw_bb(new_img, test_annot,'testtesttest'+input_img_file.split("/")[-1].split(".")[0])

                # save the img and the new annot file
                new_name = input_img_file.split("/")[-3] + "_" + input_img_file.split("/")[-1]
                new_name = new_name.replace(".png", "_"+str(m)+".png")
                output_img_file = os.path.join(current_output, "images") + "/" + new_name
                output_annot_file = os.path.join(current_output, "labels") + "/" + new_name.replace(".png", ".txt")
                write_image(new_img, output_img_file)
                write_annot([obj_list, new_annot_list], output_annot_file)


def create_yolo_dataset_2(dir_path_input, dir_path_output, seed, desired_size):
    '''
    This function generates the dataset for each object(tool) seperately
    by finding the largest bounding box in the original image
    '''

    if not os.path.exists(dir_path_output):
        os.makedirs(dir_path_output, exist_ok=True)

    all_file_list = make_list(dir_path_input)
    random.seed(seed)
    random.shuffle(all_file_list)
    #all_file_list = all_file_list[:10]
    train_test_valid_prob = [0.8, 0.2, 0.0]
    n = len(all_file_list)

    indx = [int(p*n) for p in train_test_valid_prob]
    indx_list = [[0,indx[0]],[indx[0],indx[0]+indx[1]],[indx[0]+indx[1],n]]
    mode_dict = {"train": indx_list[0], "valid": indx_list[1], "test": indx_list[2]}
    print(indx_list)

    for mode, indx_range in mode_dict.items():
        current_output = os.path.join(dir_path_output, mode)
        os.makedirs(current_output, exist_ok=True)
        data_list = all_file_list[indx_range[0]:indx_range[1]]
        os.makedirs(os.path.join(current_output,"images"), exist_ok=True)
        os.makedirs(os.path.join(current_output,"labels"), exist_ok=True)
        for item in data_list:
            input_img_file, input_annot_file = item[0], item[1]
            # open image and annot files
            print(input_img_file)
            img = read_image(input_img_file)
            (img_height, img_width, _) = np.shape(img)
            annot_list = read_annot(input_annot_file)
            print(annot_list)
            obj_list = [bb[0] for bb in annot_list]
            bb_list = [bb[1] for bb in annot_list]
            print(obj_list)

            #print("***********************")
            #print(obj_id, annot)
            print("#################" , bb_list)
            # add padding to bb list (Optional)
            #list_obj_id_with_padding = [0,1,2,3]
            #for i in range(len(obj_list)):
            #    if obj_list[i] in list_obj_id_with_padding:
            #        bb_list[i] = bb_util.add_padding_yolo_bb(bb_list[i], [img_width, img_height],pad=50)
            #print("#################" , bb_list)
            # convert yolo annot to abs annot
            bb_list = bb_util.convert_yolo_bb_to_abs(bb_list, [img_width, img_height])
            #print(bb_list)
            #exit(0)
            # find the largest bbox contour including all bboxes
            bbox_l = bb_util.find_largest_contour(bb_list)
            #print(bbox_l)
            if bbox_l[2]>desired_size[0] or bbox_l[3]>desired_size[1]:
                continue

            # crop image and get new annot
            #print(obj_id, annot)
            #new_img, new_annot_list = img_util.rnd_crop_img(img, bbox_l, bb_list, desired_size=desired_size)
            for i in range(len(obj_list)):

                for m in range(5):
                    #m=4
                    new_img, new_annot_list = img_util.rnd_crop_img_bb_side_fit(img, bb_list[i], [bb_list[i]], desired_size=desired_size, stride=100, mode=m)
                    #new_img, new_annot_list = img_util.rnd_crop_img(img, bbox_l, bb_list, desired_size=desired_size)

                    #print(new_annot_list)
                    #print(desired_size)
                    # convert back the abs annot to yolo annot
                    new_annot_list = bb_util.convert_abs_bb_to_yolo(new_annot_list, img_shape=[np.shape(new_img)[1], np.shape(new_img)[0]])
                    #test_annot = bb_util.convert_yolo_bb_to_abs(new_annot,desired_size)
                    #print(obj_list, new_annot_list)

                    #print(obj_id, test_annot)
                    #bb_util.draw_bb(new_img, test_annot,'testtesttest'+input_img_file.split("/")[-1].split(".")[0])

                    # save the img and the new annot file
                    new_name = input_img_file.split("/")[-3] + "_" + input_img_file.split("/")[-1]
                    new_name = new_name.replace(".png", "_"+str(m)+"_"+str(i)+".png")
                    output_img_file = os.path.join(current_output, "images") + "/" + new_name
                    output_annot_file = os.path.join(current_output, "labels") + "/" + new_name.replace(".png", ".txt")
                    write_image(new_img, output_img_file)
                    write_annot([[obj_list[i]], new_annot_list], output_annot_file)
                    #exit(0)
            #exit(0)


def create_yolo_dataset_multi(dir_path_input, dir_path_output, seed, desired_size):
    '''
    This function generates the dataset for pair of objects(tools) seperately
    by finding the largest bounding box between two pairs in the original image
    '''

    if not os.path.exists(dir_path_output):
        os.makedirs(dir_path_output, exist_ok=True)

    all_file_list = make_list(dir_path_input)
    random.seed(seed)
    random.shuffle(all_file_list)
    #all_file_list = all_file_list[:10]
    train_test_valid_prob = [0.9, 0.1, 0.0]
    n = len(all_file_list)
    print(n)

    indx = [int(p*n) for p in train_test_valid_prob]
    indx_list = [[0,indx[0]],[indx[0],indx[0]+indx[1]],[indx[0]+indx[1],n]]
    mode_dict = {"train": indx_list[0], "valid": indx_list[1], "test": indx_list[2]}
    print(indx_list)
    print(mode_dict)

    for mode, indx_range in mode_dict.items():
        current_output = os.path.join(dir_path_output, mode)
        os.makedirs(current_output, exist_ok=True)
        data_list = all_file_list[indx_range[0]:indx_range[1]]
        os.makedirs(os.path.join(current_output,"images"), exist_ok=True)
        os.makedirs(os.path.join(current_output,"labels"), exist_ok=True)
        for item in data_list:
            input_img_file, input_annot_file = item[0], item[1]
            # open image and annot files
            print(input_img_file)
            img = read_image(input_img_file)
            (img_height, img_width, _) = np.shape(img)
            annot_list = read_annot(input_annot_file)
            #print(annot_list)
            obj_list = [bb[0] for bb in annot_list]
            bb_list = [bb[1] for bb in annot_list]
            #print(obj_list)

            #print("***********************")
            #print(obj_id, annot)
            #print("#################" , bb_list)
            # add padding to bb list (Optional)
            #list_obj_id_with_padding = [0,1,2,3]
            #for i in range(len(obj_list)):
            #    if obj_list[i] in list_obj_id_with_padding:
            #        bb_list[i] = bb_util.add_padding_yolo_bb(bb_list[i], [img_width, img_height],pad=50)
            #print("#################" , bb_list)
            # convert yolo annot to abs annot
            bb_list = bb_util.convert_yolo_bb_to_abs(bb_list, [img_width, img_height])
            #print(bb_list)
            #exit(0)
            # find the largest bbox contour including all bboxes
            #bbox_l = bb_util.find_largest_contour(bb_list)
            #print(bbox_l)
            #if bbox_l[2]>desired_size[0] or bbox_l[3]>desired_size[1]:
            #    continue

            # crop image and get new annot
            #print(obj_id, annot)
            #new_img, new_annot_list = img_util.rnd_crop_img(img, bbox_l, bb_list, desired_size=desired_size)
            for i in range(len(obj_list)):
                #print("#######################     ", obj_list)
                #print("#######################     " , obj_list[i])
                candid_obj_list =  obj_list[i+1:]
                #candid_obj_list.remove(obj_list[:i+1])
                candid_bb_list =  bb_list[i+1:]
                #print(candid_obj_list , candid_bb_list)
                #candid_bb_list.remove(bb_list[:i+1])
                for j in range(len(candid_obj_list)):
                    bbox_l = bb_util.find_largest_contour([bb_list[i], candid_bb_list[j]])
                    #print(bbox_l)
                    if bbox_l[2] > desired_size[0] or bbox_l[3] > desired_size[1]:
                        continue
                    else:
                        final_obj_bb = [(obj_list[z],bb_list[z]) for z in range(len(obj_list)) if bb_util.bb_intersection(bbox_l[:], bb_list[z][:])]
                        final_obj_list = [obj_bb[0] for obj_bb in final_obj_bb]
                        final_bb_list = [obj_bb[1] for obj_bb in final_obj_bb]
                        for m in range(5):
                            #m=4
                            #print("§§§§§§§§§§§§§§§§", final_obj_bb)
                            #print("§§§§§§§§§§§§§§§§" , bbox_l)
                            #print("§§§§§§§§§§§§§§§§", final_bb_list)
                            new_img, new_annot_list = img_util.rnd_crop_img_bb_side_fit(img, bbox_l, final_bb_list, desired_size=desired_size, stride=20, mode=m)
                            #new_img, new_annot_list = img_util.rnd_crop_img(img, bbox_l, bb_list, desired_size=desired_size)

                            #print(new_annot_list)
                            #print(desired_size)
                            #print(new_annot_list)
                            # convert back the abs annot to yolo annot
                            new_annot_list = bb_util.convert_abs_bb_to_yolo(new_annot_list, img_shape=[np.shape(new_img)[1], np.shape(new_img)[0]])
                            #test_annot = bb_util.convert_yolo_bb_to_abs(new_annot,desired_size)
                            #print(obj_list, new_annot_list)
                            #print(new_annot_list)
                            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                            #print(obj_id, test_annot)
                            #bb_util.draw_bb(new_img, test_annot,'testtesttest'+input_img_file.split("/")[-1].split(".")[0])

                            # save the img and the new annot file
                            new_name = input_img_file.split("/")[-3] + "_" + input_img_file.split("/")[-1]
                            new_name = new_name.replace(".png", "_"+str(i)+"_"+str(j)+"_"+str(m)+".png")
                            output_img_file = os.path.join(current_output, "images") + "/" + new_name
                            output_annot_file = os.path.join(current_output, "labels") + "/" + new_name.replace(".png", ".txt")
                            write_image(new_img, output_img_file)
                            write_annot([final_obj_list, new_annot_list], output_annot_file)
                    #exit(0)
            #exit(0)

