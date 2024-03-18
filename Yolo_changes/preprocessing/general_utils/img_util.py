import os
import cv2
import numpy as np
#from skimage.metrics import structural_similarity

def read_image(img_file):
    '''
    This function reads the image file using cv2
    img_file: path to the image file
    return : the image
    '''
    img = cv2.imread(img_file)
    return img

def write_image(img, img_file):
    '''
    This function saves the image file using cv2
    img : the image
    img_file: path to the output image file
    '''
    cv2.imwrite(img_file, img)

def rotate_img_around_center(img,angle=10,scale=1):
    '''
    This function rotates the image around the center of the image by the given angle
    img : the input image
    angle : given angle
    scale : given scale
    return : the new rotated image
    '''
    (height, width, _) = np.shape(img)
    image_center = (int(width / 2), int(height / 2))
    M = cv2.getRotationMatrix2D(image_center, angle, scale)
    rotated_img = cv2.warpAffine(img, M, (width, height))
    return rotated_img

def resize_img(img, des_size=[600, 800]):
    '''
    This function resizes the image by a desired size
    img : the input image
    des_size : the desired size
    return : the new resized image
    '''
    resized = cv2.resize(img, des_size, interpolation=cv2.INTER_AREA)
    return resized

def resize_img_scale(img, scale_factor):
    '''
    This function resizes the image by a scale factor
    img : the input image
    scale_factor : the scale factor in percentage
    return : the new resized image
    '''
    width = int(np.shape(img)[1] * scale_factor / 100)
    height = int(np.shape(img)[0] * scale_factor / 100)
    new_dim = (width, height)
    resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    return resized


'''
def frames_similarity(img1, img2, scale_percent=1):
    img1_gray = cv2.cvtColor(scale_image(img1,scale_percent=scale_percent*100), cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(scale_image(img2,scale_percent=scale_percent*100), cv2.COLOR_BGR2GRAY)
    #(score, diff) = structural_similarity(img1_gray, img2_gray, full=True)
    score = structural_similarity(img1_gray, img2_gray, full=False)
    #diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))
    #cv2.imwrite("5_output_frame_diff.jpg", diff)
    #cv2.imwrite("5_output_img1.jpg", scale_image(img1,scale_percent=scale_percent*100))
    #cv2.imwrite("5_output_img2.jpg", scale_image(img2,scale_percent=scale_percent*100))
    return score
'''

def color_perturbation(img, mu=0.0, sigma=10.0):
    '''
    This function changes the color of the image by adding a random value to each pixel
    img : the input image
    mu : the mean value of the normal distribution
    sigma : the standard deviation of the normal distribution
    return : the new color perturbed image
    '''
    img = img.astype('float64')
    cp = np.random.normal(mu, sigma, 3)
    img += cp
    img[img > 255.0] = 255.0
    img[img < 0] = 0
    return img


def brightness_perturbation(img, mu=1.0, sigma=1.0):
    '''
    This function changes the brightness of the image by multiplying a random value to each pixel
    img : the input image
    mu : the mean value of the normal distribution
    sigma : the standard deviation of the normal distribution
    return : the new brightness perturbed image
    '''
    bp = -1.0
    while bp < 0:
        bp = np.random.normal(mu, sigma, 1)
    img *= bp
    img[img > 255.0] = 255.0
    img[img < 0] = 0
    return img

def rnd_circle_blood_perturbation(img,max_b_drop=10):
    '''
    This function adds random circle shaped blood drops to the image
    img : the input image
    m_b_drop : the maximum number of blood drops
    return : the new bloody image
    '''
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
    '''
    This function adds random oval blood drops to the image
    img : the input image
    m_b_drop : the maximum number of blood drops
    return : the new bloody image
    '''
    (height, width, _) = np.shape(img)
    thickness = -1
    for i in range(max_b_drop):
        blood_color = (0, 0, np.random.randint(low=100, high=255))
        angle = np.random.randint(low=0, high=360)
        axesLength = (np.random.randint(low=1, high=20), np.random.randint(low=1, high=20))
        (startAngle, endAngle) = (np.random.randint(low=0, high=1), np.random.randint(low=359, high=360))
        cx = np.random.randint(low=0, high=width-1)
        cy = np.random.randint(low=0, high=height-1)
        center_coordinates = (cx, cy)
        overlay = img.copy()
        cv2.ellipse(overlay, center_coordinates, axesLength, angle,
                          startAngle, endAngle, blood_color, thickness)
        alpha = np.random.random_sample()
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img


def crop_image_with_padding(img, area_bbox, pad_size=5):
    '''
    This function crops the image from any given point by a given pad size
    img : the input image
    bbox : the given area box in [P1x, P1y, P2x, P2y] format to crop the image
    pad_size : the given padding size
    return : the new cropped image
    '''
    p1x , p1y = max(0,area_bbox[0]-pad_size) , max(0,area_bbox[1]-pad_size)
    p2x , p2y = min(area_bbox[2]+pad_size,img.shape[1]) , min(area_bbox[3]+pad_size,img.shape[0])
    return img[int(p1y):int(p2y), int(p1x):int(p2x)]

def crop_image_from_center(img, des_size):
    '''
    This function crops the image from the center point of the image
    img : the input image
    des_size : the desired size of the cropped area
    return : the new cropped image from center
    '''
    (width, height) = des_size
    im_cx , im_cy = img.shape[1]/2 , img.shape[0]/2
    p1x , p1y = im_cx - width/2 , im_cy - height/2
    p2x , p2y = im_cx + width/2 , im_cy + height/2
    return img[int(p1y):int(p2y), int(p1x):int(p2x)]


def crop_image(img, bbox, des_size):
    '''
    This function crops the image around the given bounding box
    img : the input image
    bbox : the given bounding box in [x, y, w, h] format
    des_size : the desired size of the cropped area
    return : the new cropped image
    '''
    (img_height, img_width, _) = np.shape(img)
    [d_width, d_height] = des_size
    bb_cx, bb_cy, bb_w, bb_h = bbox[0], bbox[1], bbox[2], bbox[3]
    if (d_height < bb_h) or (d_width < bb_w):
        return (img, bbox)
    else:
        img = img[bb_cy:bb_cy+d_height, bb_cx:bb_cx+d_width, :]
        return img

def rnd_crop_image(img, bbox, bbox_list, des_size):
    '''
    This function crops the image randomly where the bounding box is inside the cropped area
    img : the input image
    bbox : the given bounding box in [x, y, w, h] format
    bbox_list : the list of bounding boxes inside the image
    des_size : the desired size of the cropped area
    return : tuple (the new cropped image , the new bounding boxes list)
    '''
    (img_height, img_width, _) = np.shape(img)
    [d_width, d_height] = des_size
    bb_cx, bb_cy, bb_w, bb_h = bbox[0], bbox[1], bbox[2], bbox[3]
    if (d_height < bb_h) or (d_width < bb_w):
        return (img, bbox)
    else:
        flag = 0
        while not flag:
            (crop_x, crop_y) = (np.random.randint(low=0, high=bb_cx), np.random.randint(low=0, high=bb_cy))
            if (crop_y+d_height > bb_cy+bb_h) and (crop_x+d_width > bb_cx+bb_w):
                if (crop_y+d_height < img_height) and (crop_x+d_width < img_width):
                    flag = 1
        img = img[crop_y:crop_y+d_height, crop_x:crop_x+d_width, :]
        new_bbox_list = [[bb[0] - crop_x, bb[1] - crop_y, bb[2], bb[3]] for bb in bbox_list]
        return (img, new_bbox_list)


def rnd_crop_image_bb_fit(img, bbox, bbox_list, pad_size=100):
    '''
    This function crops the image randomly fitting the bounding box inside the cropped area with a padding
    img : the input image
    bbox : the given bounding box in [x, y, w, h] format
    bbox_list : the list of bounding boxes inside the image
    pad_size : the desired constant padding size
    return : tuple (the new cropped image , the new bounding boxes list)
    '''
    (img_height, img_width, _) = np.shape(img)
    bb_cx, bb_cy, bb_w, bb_h = bbox[0], bbox[1], bbox[2], bbox[3]
    d_width, d_height = int(bb_w+pad_size), int(bb_h+pad_size)
    if (d_height < bb_h) or (d_width < bb_w):
        return (img, bbox)
    else:
        flag = 0
        while not flag:
            (crop_x, crop_y) = (np.random.randint(low=bb_cx-100, high=bb_cx), np.random.randint(low=bb_cy-100, high=bb_cy))
            if (crop_y+d_height > bb_cy+bb_h) and (crop_x+d_width > bb_cx+bb_w):
                if (crop_y+d_height < img_height) and (crop_x+d_width < img_width):
                    flag = 1
        img = img[crop_y:crop_y+d_height, crop_x:crop_x+d_width, :]
        new_bbox_list = [[bb[0] - crop_x, bb[1] - crop_y, bb[2], bb[3]] for bb in bbox_list]
        return (img, new_bbox_list)


def rnd_crop_image_bb_side_fit(img, bbox, bbox_list, pad_size=20, max_try=10, mode=0):
    '''
    This function crops the image randomly fitting the bounding box inside the cropped area with a padding
    where the bounding box is fitted to the left, right, top, bottom or center of the cropped area
    img : the input image
    bbox : the given bounding box in [x, y, w, h] format
    bbox_list : the list of bounding boxes inside the image
    pad_size : the desired constant padding size
    max_try : the maximum number of trials for fitting the bounding box
    mode : the mode of fitting the bounding box to the cropped area (0:left, 1:right, 2:top, 3:bottom, 4:center)
    return : tuple (the new cropped image , the new bounding boxes list)
    '''
    (img_height, img_width, _) = np.shape(img)
    bb_cx, bb_cy, bb_w, bb_h = bbox[0], bbox[1], bbox[2], bbox[3]
    d_width, d_height = int(bb_w + pad_size), int(bb_h + pad_size)
    if (d_height < bb_h) or (d_width < bb_w):
        return (img, bbox)
    else:
        flag = 0
        num_try = 0
        while not flag and num_try < max_try:
            if mode == 0: #left-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - (0.25 * pad_size), high=bb_cx - 2),
                    np.random.randint(low=bb_cy - pad_size, high=bb_cy))
            elif mode == 1: #right-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - pad_size + 2, high=bb_cx - (0.75 * pad_size)),
                    np.random.randint(low=bb_cy - pad_size, high=bb_cy))
            elif mode == 2:  # top-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - pad_size, high=bb_cx),
                    np.random.randint(low=bb_cy - (0.25 * pad_size), high=bb_cy - 2))
            elif mode == 3:  # bottom-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - pad_size, high=bb_cx),
                    np.random.randint(low=bb_cy - pad_size + 2, high=bb_cy - (0.75 * pad_size)))
            elif mode == 4:  # center-fit
                (crop_x, crop_y) = (
                    np.random.randint(low=bb_cx - (0.5 * pad_size) - 2, high=bb_cx - (0.5 * pad_size) + 2),
                    np.random.randint(low=bb_cy - (0.5 * pad_size) - 2, high=bb_cy - (0.5 * pad_size) + 2))
            if (bb_cy+bb_h <= crop_y+d_height < img_height) and (bb_cx+bb_w <= crop_x+d_width < img_width) :
                flag = 1
            else:
                num_try += 1
            
        if flag :
            img = img[crop_y:crop_y+d_height, crop_x:crop_x+d_width, :]
            new_bbox_list = [[bb[0] - crop_x, bb[1] - crop_y, bb[2], bb[3]] for bb in bbox_list]
        else :
            img = None
            new_bbox_list = None
        return (img, new_bbox_list)


def convert_frames_to_vid(frames_path, des_size=[960, 960], fps=5):
    '''
    This function converts the frames to video(.avi)
    frames_path : the input frames path
    des_size : the desired size of the video
    fps : the frame per second rate
    '''
    frames_list = []
    for root, dirnames, filenames in os.walk(frames_path):
        filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
        for filename in filenames:
            if filename.endswith(('.png', '.PNG')):
                print(filename)
                frames_list.append(os.path.join(root, filename))

    vid_output_path = frames_path.replace(frames_path.split("/")[-1].split("_")[-1], "video")
    os.makedirs(vid_output_path, exist_ok=True)
    vid_name = vid_output_path.split("/")[-1] + ".avi"

    out = cv2.VideoWriter(os.path.join(vid_output_path, vid_name), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                          tuple(des_size))
    for frame_path in frames_list:
        img = read_image(frame_path)
        img = cv2.resize(img, tuple(des_size), interpolation=cv2.INTER_AREA)
        out.write(img)

    out.release()
    print("Video has been generated...", os.path.join(vid_output_path, vid_name))


def convert_vid_to_frames(vid_path, des_size=[960, 960], fps=5):
    '''
    This function converts the video to frames(.png)
    vid_path : the input video path
    des_size : the desired size of the frames
    fps : the frame per second rate
    '''
    vid_name = vid_path.split("/")[-1].split(".")[0]
    frame_output_path = os.path.join(vid_path.replace(vid_path.split("/")[-1], ""), vid_name+"_frames")
    os.makedirs(frame_output_path, exist_ok=True)
    vidcap = cv2.VideoCapture(vid_path)
    sec = 0
    count = 1
    success = True
    while success:
        count = count + 1
        sec = sec + 1/fps
        sec = round(sec, 2)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, frame = vidcap.read()
        if hasFrames:
            # manually crop the frame
            #frame = crop_image(frame, [500, 100, 960, 960], [960, 960])
            # resize the frame to the desired size
            frame = resize_img(frame, des_size)
            cv2.imwrite(os.path.join(frame_output_path, vid_name+"_frame_" + str(count).zfill(5) + ".png"), frame)
        else:
            success = False

