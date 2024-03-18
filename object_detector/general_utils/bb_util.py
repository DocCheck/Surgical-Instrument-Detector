import math
import numpy as np


def convert_yolo_bb_to_abs(annot_list,img_shape):
    #print(annot)
    #print(len(annot))

    [width, height] = img_shape
    abs_annot_list = []
    for annot in annot_list:
        abs_annot = [0,0,0,0]
        abs_annot[0] = (annot[0] * width) - ((annot[2] * width) / 2)
        abs_annot[1] = (annot[1] * height) - ((annot[3] * height) / 2)
        abs_annot[2] = annot[2] * width
        abs_annot[3] = annot[3] * height
        abs_annot_list.append(abs_annot)
    return abs_annot_list

def convert_abs_bb_to_yolo(annot_list,img_shape):

    [width, height] = img_shape
    yolo_annot_list = []
    for annot in annot_list:
        b_width, b_height = annot[2], annot[3]
        x_max = annot[0] + b_width
        y_max = annot[1] + b_height
        yolo_annot = [0,0,0,0]
        yolo_annot[0] = ((x_max + annot[0]) / 2) / width
        yolo_annot[1] = ((y_max + annot[1]) / 2) / height
        yolo_annot[2] = b_width / width
        yolo_annot[3] = b_height / height
        yolo_annot_list.append(yolo_annot)

    return yolo_annot_list

def convert_bb_xywh(bbox_list):
    # input x1y1x2y2 output xywh
    bbox_xywh = [[box[0],box[1],box[2]-box[0],box[3]-box[1]] for box in bbox_list]
    return bbox_xywh

def bb_center_coord(bbox_list):
    # input x1y1x2y2
    bbox_center = [[box[2]-box[0]/2,box[3]-box[1]/2] for box in bbox_list]
    return bbox_center

def find_largest_contour(bb_list):
    min_x = min([x[0] for x in bb_list])
    min_y = min([x[1] for x in bb_list])
    max_x = max([x[0] + x[2] for x in bb_list])
    max_y = max([x[1] + x[3] for x in bb_list])
    return [min_x, min_y, max_x - min_x, max_y - min_y]

def bb_intersection(boxA, boxB):
    #print(boxA,boxB)
    boxAArea = round(boxA[2] * boxA[3] , 2)
    boxBArea = round(boxB[2] * boxB[3] , 2)
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA[2],boxA[3] = boxA[0]+boxA[2] , boxA[1]+boxA[3]
    boxB[2],boxB[3] = boxB[0]+boxB[2] , boxB[1]+boxB[3]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = round(max(abs(xB - xA), 0) * max(abs(yB - yA), 0) , 2)
    #print(interArea, boxAArea, boxBArea)
    if interArea == boxAArea or interArea == boxBArea:
        return True
    else :
        return False


def add_padding_yolo_bb(annot,img_shape,pad=4):
    [width, height] = img_shape
    padded_annot = [annot[0], annot[1], 0, 0]
    padded_annot[2] = annot[2]  + (pad / width)
    padded_annot[3] = annot[3]  + (pad / height)
    return padded_annot


def rotate_point(origin, point, angle):
    [ox, oy] = origin
    [px, py] = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def rotate_bb_around_center(img, bbox, angle=10, scale=1):
    (height, width, _) = np.shape(img)
    cx, cy = (int(width / 2), int(height / 2))

    # bbox_8 = [bbox[0],bbox[1]+bbox[3],bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
    bbox_4 = [[bbox[0], bbox[1]], [bbox[0], bbox[1] + bbox[3]], [bbox[0] + bbox[2], bbox[1] + bbox[3]],
              [bbox[0] + bbox[2],
               bbox[1]]]
    print("***********", bbox_4)

    rectangle = [bbox_4[0], bbox_4[1], bbox_4[2], bbox_4[3], bbox_4[0]]
    rectangle_rotated = [rotate_point([cx, cy], pt, math.radians(angle)) for pt in rectangle]

    rectangle = np.array(rectangle)
    rectangle_rotated = np.array(rectangle_rotated)
    # these are what you need
    x_min, y_min = np.min(rectangle_rotated, axis=0)
    x_max, y_max = np.max(rectangle_rotated, axis=0)
    # create bounding rect points
    pt1 = [x_min, y_min]
    pt2 = [x_min, y_max]
    pt3 = [x_max, y_max]
    pt4 = [x_max, y_min]

    #rectangle_bounding = [pt1, pt2, pt3, pt4, pt1]
    rotated_annot = [pt1[0], pt1[1], pt3[0] - pt1[0], pt2[1] - pt1[1]]
    print(rotated_annot)
    return rotated_annot


def resize_bb(bbox, scale_factor):
    # resize bbox
    bbox = np.array(bbox) * scale_factor / 100
    print('BBox : ', bbox.tolist())
    return bbox.tolist()


def draw_bb(img,bb,output_name):
    import cv2
    #img = cv2.imread(img_file)
    print(np.shape(img))
    bb = [int(x) for x in bb]
    cv2.rectangle(img, (bb[0],bb[1]), (bb[0]+bb[2],bb[1]+bb[3]), color=(0, 255, 0), thickness=2)
    cv2.imwrite("output_data/test/alone1_1_"+str(output_name)+".jpg", img)

