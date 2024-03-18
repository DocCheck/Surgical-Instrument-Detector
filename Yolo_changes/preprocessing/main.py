import os
import cv2
import sys
from preprocessing.general_utils import img_util

import argparse


def parse_opt(parser, known=False):
    parser.add_argument("--output-path", type=str, default="data/dataset/captured_samples/",
                        help="Captured frames output path folder")
    parser.add_argument("--obj-name", type=str, default="sample_frame", help="Object/Sample frame name")
    parser.add_argument("--frames-path", type=str, default="data/dataset/captured_samples/frames/sample_frame/",
                        help="Frames path folder to convert to video")
    parser.add_argument("--video-name", type=str, default="sample_video.avi", help="Video name")
    parser.add_argument("--video-path", type=str, default="data/dataset/captured_samples/videos/sample_video.avi",
                        help="video path to convert to frames")
    parser.add_argument("--video-res", type=int, nargs='+', default=[960, 960], help="Video resolution w/h")
    parser.add_argument("--fps", type=int, default=5, help="Captured video frame per second")
    parser.add_argument("--cam-port", type=int, default=0, help="Camera port")
    parser.add_argument("--cam-res", type=int, nargs='+', default=[1920, 1080], help="Captured frame resolution w/h")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def cam_rec_img(opt):
    final_path = os.path.join(opt.output_path, "frames", opt.obj_name)
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    cam = cv2.VideoCapture(opt.cam_port)
    print("Camera port is : ", opt.cam_port)
    print("Capturing frame resolution is : ", opt.cam_res)
    cam.set(3, opt.cam_res[0])
    cam.set(4, opt.cam_res[1])

    # If image will be detected without any error, show result
    img_counter = 1
    while (True):

        # Capture the video frame by frame
        ret, frame = cam.read()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # the 'q' button is set as the quitting button.
        # the 's' button is set as the saving button.
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        if k & 0xFF == ord('s'):
            img_name = os.path.join(final_path, opt.obj_name + "_" + str(img_counter).zfill(5) + ".png")
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    # After the loop release the cap object
    cam.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def cam_rec_video(opt):
    final_path = os.path.join(opt.output_path, "videos")
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    cam = cv2.VideoCapture(opt.cam_port)
    print("Capturing video resolution is : ", opt.cam_res)
    cam.set(3, opt.cam_res[0])
    cam.set(4, opt.cam_res[1])

    vid_writer = cv2.VideoWriter(os.path.join(final_path, opt.video_name), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                 opt.fps, tuple(opt.cam_res))

    # If image will be detected without any error, show result
    record = False
    while (True):

        # Capture the video frame by frame
        ret, frame = cam.read()
        # the 's' button is set as starting to record button.
        # the 'q' button is set as stopping and quitting button.
        k = cv2.waitKey(1)
        if ret == True:
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if k & 0xFF == ord('s'):
                record = True
            if record:
                vid_writer.write(frame)
                print('frame saved')
            if k & 0xFF == ord('q'):
                break
        else:
            break

    # After the loop release the cap object
    cam.release()
    vid_writer.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def extract_frames_from_vid(opt):
    img_util.convert_vid_to_frames(opt.video_path, des_size=opt.video_res, fps=opt.fps)


def convert_frames_to_vid(opt):
    img_util.convert_frames_to_vid(opt.frames_path, des_size=opt.video_res, fps=opt.fps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    action = sys.argv[-1]
    opt = parse_opt(parser)

    if action == "camera-capture-frames":
        cam_rec_img(opt)
    elif action == "camera-capture-video":
        cam_rec_video(opt)
    elif action == "convert-video-frames":
        extract_frames_from_vid(opt)
    elif action == "convert-frames-video":
        convert_frames_to_vid(opt)
    else:
        print("WRONG ACTION")
        exit(1)
