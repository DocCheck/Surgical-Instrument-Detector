import torch
import numpy as np
from instrument_detector.object_detector.Rona_mid.utils.general import non_max_suppression, scale_boxes, imread
from instrument_detector.object_detector.Rona_mid.utils.augmentations import letterbox
from instrument_detector.object_detector.Rona_mid.models.experimental import Ensemble


class Rona_mid:
    def __init__(self, model_config):
        self.model_path = model_config["model_path"]
        self.imgsz = model_config["image_size"]
        self.conf_thres = model_config["confidence_threshold"]
        self.iou_thres = model_config["iou_threshold"]
        self.classes = model_config["classes"]
        self.agnostic_nms = model_config["agnostic_nms"]
        self.max_det = model_config["maximum_detection"]
        self.device = torch.device(model_config["device"])


    def load_model(self):
        self.model = Ensemble()
        ckpt = torch.load(self.model_path, map_location=self.device)
        ckpt = (ckpt.get('ema') or ckpt['model']).to(self.device).float()  # FP32 model
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        #self.model.append(ckpt.eval())  # model in eval mode
        self.model = ckpt.eval()
        self.stride = max(int(ckpt.stride.max()), 32)  # model stride
        self.names = ckpt.module.names if hasattr(ckpt, 'module') else ckpt.names
        #print(self.stride)
        #print(self.names)
        return self.model, self.names


    def predict(self, img_path):
        if len(np.shape(img_path))==3:
            # input is already an image
            im0 = img_path
        else:
            # read the input image
            im0 = imread(img_path)

        # pre-processing the image
        # resize the image with padding
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=True, scaleup=False)[0]  # padded resize
        #imwrite("sample_im0.jpg",im0)
        #imwrite("sample_im.jpg",im)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        #print(pred)
        # scale back the predicted boxes to the orig img size
        for i, det in enumerate(pred):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        #print(pred)
        return pred

