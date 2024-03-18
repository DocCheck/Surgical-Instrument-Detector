import os
import sys
import cv2
import numpy as np
import math


def read_image(img_file):
    img = cv2.imread(img_file)
    return img

def rotate_img_around_center(img,angle=10,scale=1):
    (height, width, _) = np.shape(img)
    image_center = (int(width / 2), int(height / 2))
    M = cv2.getRotationMatrix2D(image_center, angle, scale)

    rotated_img = cv2.warpAffine(img, M, (width, height))
    cv2.imwrite("output_data/test/alone1_1_rotated.jpg", rotated_img)
    return rotated_img

def resize_img(img, scale_factor):
    #scale_percent = 60  # percent of original size
    width = int(np.shape(img)[1] * scale_factor / 100)
    height = int(np.shape(img)[0] * scale_factor / 100)
    new_dim = (width, height)
    # resize image
    resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    return resized


def crop_image(im, bbox, pad_size=5):
    p1x , p1y = max(0,bbox[0]-pad_size) , max(0,bbox[1]-pad_size)
    p2x , p2y = min(bbox[2]+pad_size,im.shape[1]) , min(bbox[3]+pad_size,im.shape[0])
    return im[int(p1y):int(p2y), int(p1x):int(p2x)]

def crop_image_from_center(im, shape):
    (width, height) = shape
    im_cx , im_cy = im.shape[1]/2 , im.shape[0]/2
    p1x , p1y = im_cx - width/2 , im_cy - height/2
    p2x , p2y = im_cx + width/2 , im_cy + height/2
    return im[int(p1y):int(p2y), int(p1x):int(p2x)]

def scale_image(im, scale_percent):
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    resized = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def color_perturbation(img):
    img = img.astype('float64')
    mu, sigma = 0.0, 10.0 #50.0
    cp = np.random.normal(mu, sigma, 3)
    print("CP : --------------- ",cp)
    img += cp
    img[img > 255.0] = 255.0
    img[img < 0] = 0
    return img


def brightness_perturbation(img):
    mu, sigma = 1.0, 1.0
    bp = -1.0
    while bp < 0:
        bp = np.random.normal(mu, sigma, 1)
    print("BP : --------------- ",bp)
    img *= bp
    img[img > 255.0] = 255.0
    img[img < 0] = 0
    return img

def rnd_circle_blood_perturbation(img,max_b_drop=10):
    (height, width, _) = np.shape(img)

    thickness = -1
    for i in range(max_b_drop):
        blood_color = (0, 0, np.random.randint(low=100, high=255))
        radius = np.random.randint(low=1, high=20)
        cx = np.random.randint(low=0, high=width-1)
        cy = np.random.randint(low=0, high=height-1)
        center_coordinates = (cx, cy)
        img = cv2.circle(img, center_coordinates, radius, blood_color, thickness)

    return img

def rnd_oval_blood_perturbation(img,max_b_drop=10):
    (height, width, _) = np.shape(img)

    thickness = -1
    for i in range(max_b_drop):
        blood_color = (0, 0, np.random.randint(low=100, high=255))
        angle = np.random.randint(low=0, high=360)
        axesLength = (np.random.randint(low=1, high=20), np.random.randint(low=1, high=20))
        (startAngle, endAngle) = (np.random.randint(low=0, high=1), np.random.randint(low=359, high=360))
        #radius = np.random.randint(low=1, high=20)
        cx = np.random.randint(low=0, high=width-1)
        cy = np.random.randint(low=0, high=height-1)
        center_coordinates = (cx, cy)
        overlay = img.copy()
        cv2.ellipse(overlay, center_coordinates, axesLength, angle,
                          startAngle, endAngle, blood_color, thickness)
        alpha = np.random.random_sample()
        #alpha = 0.8
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img


def crop_img(img, bbox_l, desired_size):
    (img_height, img_width, _) = np.shape(img)
    [d_width, d_height] = desired_size
    bb_cx, bb_cy, bb_w, bb_h = bbox_l[0], bbox_l[1], bbox_l[2], bbox_l[3]
    if (d_height < bb_h) or (d_width < bb_w):
        print("#######")
        return (img, bbox_l)
    else:
        img = img[bb_cy:bb_cy+d_height, bb_cx:bb_cx+d_width, :]
        print("image_size after crop :", np.shape(img))
        return img

def rnd_crop_img(img, bbox_l, bbox_list, desired_size):
    (img_height, img_width, _) = np.shape(img)
    [d_width, d_height] = desired_size
    bb_cx, bb_cy, bb_w, bb_h = bbox_l[0], bbox_l[1], bbox_l[2], bbox_l[3]
    if (d_height < bb_h) or (d_width < bb_w):
        return (img, bbox_l)
    else:
        flag = 0
        while not flag:
            (crop_x, crop_y) = (np.random.randint(low=0, high=bb_cx), np.random.randint(low=0, high=bb_cy))
            if (crop_y+d_height > bb_cy+bb_h) and (crop_x+d_width > bb_cx+bb_w):
                if (crop_y+d_height < img_height) and (crop_x+d_width < img_width):
                    flag = 1
        img = img[crop_y:crop_y+d_height, crop_x:crop_x+d_width, :]
        print("image_size after crop :", np.shape(img))
        #new_bbox = [bb_cx-crop_x, bb_cy-crop_y, bb_w, bb_h]
        new_bbox_list = [[bb[0] - crop_x, bb[1] - crop_y, bb[2], bb[3]] for bb in bbox_list]
        print("bbox after crop :", new_bbox_list)
        return (img, new_bbox_list)


def rnd_crop_img_bb_fit(img, bbox_l, bbox_list, desired_size):
    (img_height, img_width, _) = np.shape(img)
    bb_cx, bb_cy, bb_w, bb_h = bbox_l[0], bbox_l[1], bbox_l[2], bbox_l[3]
    d_width, d_height = int(bb_w+100), int(bb_h+100)
    if (d_height < bb_h) or (d_width < bb_w):
        print("************************")
        return (img, bbox_l)
    else:
        flag = 0
        while not flag:
            (crop_x, crop_y) = (np.random.randint(low=bb_cx-100, high=bb_cx), np.random.randint(low=bb_cy-100, high=bb_cy))
            print("###############", crop_x, crop_y)
            if (crop_y+d_height > bb_cy+bb_h) and (crop_x+d_width > bb_cx+bb_w):
                if (crop_y+d_height < img_height) and (crop_x+d_width < img_width):
                    flag = 1
        img = img[crop_y:crop_y+d_height, crop_x:crop_x+d_width, :]
        print("image_size after crop :", np.shape(img))
        #new_bbox = [bb_cx-crop_x, bb_cy-crop_y, bb_w, bb_h]
        new_bbox_list = [[bb[0] - crop_x, bb[1] - crop_y, bb[2], bb[3]] for bb in bbox_list]
        print("bbox after crop :", new_bbox_list)
        return (img, new_bbox_list)


def rnd_crop_img_bb_side_fit(img, bbox_l, bbox_list, desired_size, mode=0, stride=20):
    (img_height, img_width, _) = np.shape(img)
    bb_cx, bb_cy, bb_w, bb_h = bbox_l[0], bbox_l[1], bbox_l[2], bbox_l[3]
    #print("bb_cx and bb_cy = ", bb_cx , bb_cy)
    d_width, d_height = int(bb_w + stride), int(bb_h + stride)
    if (d_height < bb_h) or (d_width < bb_w):
        #print("************************")
        return (img, bbox_l)
    else:
        flag = 0
        while not flag:
            if mode == 0: #left-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - (0.25 * stride), high=bb_cx - 2),
                    np.random.randint(low=bb_cy - stride, high=bb_cy))
            elif mode == 1: #right-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - stride + 2, high=bb_cx - (0.75 * stride)),
                    np.random.randint(low=bb_cy - stride, high=bb_cy))
            elif mode == 2:  # top-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - stride, high=bb_cx),
                    np.random.randint(low=bb_cy - (0.25 * stride), high=bb_cy - 2))
            elif mode == 3:  # bottom-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - stride, high=bb_cx),
                    np.random.randint(low=bb_cy - stride + 2, high=bb_cy - (0.75 * stride)))
            elif mode == 4:  # center-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - (0.5 * stride) - 2, high=bb_cx - (0.5 * stride) + 2),
                    np.random.randint(low=bb_cy - (0.5 * stride) - 2, high=bb_cy - (0.5 * stride) + 2))
            #print("###############", crop_x, crop_y)
            if (crop_y+d_height > bb_cy+bb_h) and (crop_x+d_width > bb_cx+bb_w):
                if (crop_y+d_height < img_height) and (crop_x+d_width < img_width):
                    flag = 1
        img = img[crop_y:crop_y+d_height, crop_x:crop_x+d_width, :]
        #print("image_size after crop :", np.shape(img))
        #new_bbox = [bb_cx-crop_x, bb_cy-crop_y, bb_w, bb_h]
        #print(bbox_list)
        new_bbox_list = [[bb[0] - crop_x, bb[1] - crop_y, bb[2], bb[3]] for bb in bbox_list]
        #print("bbox after crop :", new_bbox_list)
        return (img, new_bbox_list)



def convert_vid_to_frames(vid_file, size=(480,480), frame_rate=0.2):
    vid_name = vid_file.split("/")[-1].split(".")[0]
    frame_output_path = os.path.join(vid_file.replace(vid_file.split("/")[-1], ""), vid_name+"_frames")
    os.makedirs(frame_output_path, exist_ok=True)
    vidcap = cv2.VideoCapture(vid_file)
    sec = 0
    count = 1
    success = True
    while success:
        count = count + 1
        sec = sec + frame_rate
        sec = round(sec, 2)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, frame = vidcap.read()
        if hasFrames:
            #frame = crop_img(frame, [0, 0, 1920, 1080], [1920, 1080])
            #frame = crop_img(frame, [500, 200, 800, 600], [800, 600])
            frame = crop_img(frame, [500, 100, 960, 960], [960, 960])
            #frame = resize_img(frame, 50)
            cv2.imwrite(os.path.join(frame_output_path, vid_name+"_frame_" + str(count).zfill(5) + ".png"), frame)
        else:
            success = False


def convert_frames_to_vid(frames_path, size=(960,960), frame_rate=0.25):
    frames_list = []
    for root, dirnames, filenames in os.walk(frames_path):
        #print(os.path.splitext(filenames[0]))
        filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0].split("_")[-2]))
        for filename in filenames:
            if filename.endswith(('.png','.PNG')):
                print(filename)
                frames_list.append(os.path.join(root, filename))

    print(frames_path.split("/")[-1].split("_")[-1])
    print(frames_path.split("/")[-1])
    vid_output_path = frames_path.replace(frames_path.split("/")[-1].split("_")[-1],"video")
    print(vid_output_path)
    os.makedirs(vid_output_path, exist_ok=True)
    vid_name = vid_output_path.split("/")[-1] + ".avi"
    out = cv2.VideoWriter(os.path.join(vid_output_path,vid_name), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, size)

    for frame_path in frames_list:
        print(frame_path)
        img = cv2.imread(frame_path)
        print(np.shape(img))
        out.write(img)

    out.release()



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)