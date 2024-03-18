import math
import numpy as np


def convert_yolo_bb_to_abs(annot_list, img_shape):
    '''
    This function converts the Yolo bounding box to absolute bounding box
    annot_list : list of bounding boxes in yolo format
    img_shape : shape of the input image in [width, height]
    return : list of absolute bounding boxes
    '''
    [width, height] = img_shape
    abs_annot_list = []
    for annot in annot_list:
        abs_annot = [0, 0, 0, 0]
        abs_annot[0] = (annot[0] * width) - ((annot[2] * width) / 2)
        abs_annot[1] = (annot[1] * height) - ((annot[3] * height) / 2)
        abs_annot[2] = annot[2] * width
        abs_annot[3] = annot[3] * height
        abs_annot_list.append(abs_annot)
    return abs_annot_list


def convert_abs_bb_to_yolo(annot_list, img_shape):
    '''
    This function converts the absolute bounding box to Yolo bounding box
    annot_list : list of absolute bounding boxes
    img_shape : shape of the input image in [width, height]
    return : list of bounding boxes in yolo format
    '''
    [width, height] = img_shape
    yolo_annot_list = []
    for annot in annot_list:
        b_width, b_height = annot[2], annot[3]
        x_max = annot[0] + b_width
        y_max = annot[1] + b_height
        yolo_annot = [0, 0, 0, 0]
        yolo_annot[0] = ((x_max + annot[0]) / 2) / width
        yolo_annot[1] = ((y_max + annot[1]) / 2) / height
        yolo_annot[2] = b_width / width
        yolo_annot[3] = b_height / height
        yolo_annot_list.append(yolo_annot)
    return yolo_annot_list


def convert_bb_xywh(bbox_list):
    '''
    This function converts the X1Y1X2Y2 bounding box format to XYWH format
    bbox_list : list of bounding boxes in X1Y1X2Y2 format
    return : list of bounding boxes in XYWH format
    '''
    bbox_xywh = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in bbox_list]
    return bbox_xywh


def bb_center_coord(bbox_list):
    '''
    This function returns the center coordinates of the bounding boxes in X1Y1X2Y2 format
    bbox_list : list of bounding boxes in X1Y1X2Y2 format
    return : list of center coordinates of the bounding boxes
    '''
    bbox_center = [[box[2] - box[0] / 2, box[3] - box[1] / 2] for box in bbox_list]
    return bbox_center


def find_largest_contour(bbox_list):
    '''
    This function returns the largest bounding box contour from the list of bounding boxes
    bbox_list : list of bounding boxes
    return : largest bounding box contour
    '''
    min_x = min([x[0] for x in bbox_list])
    min_y = min([x[1] for x in bbox_list])
    max_x = max([x[0] + x[2] for x in bbox_list])
    max_y = max([x[1] + x[3] for x in bbox_list])
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def bb_intersection(boxA, boxB):
    '''
    This function finds the intersection between two bounding boxes in absolute format
    boxA : bounding box 1
    boxB : bounding box 2
    return : True if boxA is completely inside boxB or vice versa, else False
    '''
    boxAArea = round(boxA[2] * boxA[3], 2)
    boxBArea = round(boxB[2] * boxB[3], 2)
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA[2], boxA[3] = boxA[0] + boxA[2], boxA[1] + boxA[3]
    boxB[2], boxB[3] = boxB[0] + boxB[2], boxB[1] + boxB[3]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = round(max(abs(xB - xA), 0) * max(abs(yB - yA), 0), 2)
    if interArea == boxAArea or interArea == boxBArea:
        return True
    else:
        return False


def add_padding_yolo_bb(annot, img_shape, pad=4):
    '''
    This function adds extra padding to the bounding box
    annot : annotation
    img_shape : shape of the input image in [width, height]
    pad : padding size default 4
    return : padded annotation
    '''
    [width, height] = img_shape
    padded_annot = [annot[0], annot[1], 0, 0]
    padded_annot[2] = annot[2] + (pad / width)
    padded_annot[3] = annot[3] + (pad / height)
    return padded_annot


def rotate_point(origin, point, angle):
    '''
    This function rotates the point around the origin point by the given angle
    origin : point around which the point is to be rotated
    point : point to be rotated
    angle : given angle
    return : the new coordinates of the rotated point
    '''
    [ox, oy] = origin
    [px, py] = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def rotate_bb_around_center(img, bbox, angle=10):
    '''
    This function rotates the bounding box around the center of the image by the given angle
    img : the input image
    bbox : the bounding box to be rotated in [x, y, w, h] format
    angle : given angle
    return : the new rotated bounding box
    '''
    (height, width, _) = np.shape(img)
    cx, cy = (int(width / 2), int(height / 2))
    # bbox_8 = [bbox[0],bbox[1]+bbox[3],bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
    # 4 points bounding box [top left, top right, bottom right, bottom left]
    bbox_4 = [[bbox[0], bbox[1]], [bbox[0], bbox[1] + bbox[3]], [bbox[0] + bbox[2], bbox[1] + bbox[3]],
              [bbox[0] + bbox[2],
               bbox[1]]]

    rectangle = [bbox_4[0], bbox_4[1], bbox_4[2], bbox_4[3], bbox_4[0]]
    rectangle_rotated = [rotate_point([cx, cy], pt, math.radians(angle)) for pt in rectangle]
    rectangle = np.array(rectangle)
    rectangle_rotated = np.array(rectangle_rotated)

    x_min, y_min = np.min(rectangle_rotated, axis=0)
    x_max, y_max = np.max(rectangle_rotated, axis=0)
    # create bounding rect points
    pt1 = [x_min, y_min]
    pt2 = [x_min, y_max]
    pt3 = [x_max, y_max]
    pt4 = [x_max, y_min]

    rotated_annot = [pt1[0], pt1[1], pt3[0] - pt1[0], pt2[1] - pt1[1]]
    print(rotated_annot)
    return rotated_annot


def rescale_bb(bbox, scale_factor):
    '''
    This function rescales the bounding box by the given scale factor
    scale_factor : the given scale factor in percentage
    return : the new rescaled bounding box
    '''
    bbox = np.array(bbox) * scale_factor / 100
    return bbox.tolist()
