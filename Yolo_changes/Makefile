dockerContainer = rona_yolov5
dockerParams = -v $$PWD/:/project -it
dockerGPUParams = -v $$PWD/:/project -it --gpus all

ifeq ($(env), ci)
	dockerParams =
endif

default:
	@echo "hello world"

build:
	docker build -t $(dockerContainer) .


bash:
	docker run $(dockerParams) $(dockerContainer) bash


camera-capture-frames:
	docker run $(dockerParams) $(dockerContainer) python3 -m  preprocessing.main --cam-port 0 --cam-res 1920 1080 --output-path data/dataset/captured_samples/ --obj-name sample_frame camera-capture-frames

camera-capture-video:
	docker run $(dockerParams) $(dockerContainer) python3 -m  preprocessing.main --cam-port 0 --cam-res 1920 1080 --output-path data/dataset/captured_samples/ --video-name sample_video.avi --fps 5 camera-capture-video

convert-frames-video:
	docker run $(dockerParams) $(dockerContainer) python3 -m  preprocessing.main --frames-path  data/dataset/test_frames1 --video-res 600 400 --fps 5 convert-frames-video

convert-video-frames:
	docker run $(dockerParams) $(dockerContainer) python3 -m  preprocessing.main --video-path  data/dataset/test_video/test_video.avi --video-res 300 200 --fps 5 convert-video-frames

data-preparation:
	docker run $(dockerParams) $(dockerContainer) python3 -m  preprocessing.run --input-data-single data/dataset/Orig_OPBesteck_dataset_single/ --input-data-multi data/dataset/Orig_OPBesteck_dataset_multi/ --output-data-path data/dataset/Rona_dataset_final_single_multi_fit/ --bb-size 640 --padding 50 --train-val-test 0.9 0.1 0.0  --seed 42

train-image:
	docker run $(dockerGPUParams) $(dockerContainer) python3 -m  train --data data/dataset/Rona_dataset_final_single_multi_fit/data.yaml --weights yolov5l.pt --hyp rona.hyp.scratch.yaml --img 960 --epochs 200 --batch-size 16 --workers 0 --name Rona_detector_model --seed 42

predict-image:
	docker run $(dockerParams) $(dockerContainer) python3 -m  detect --weights runs/train/Rona_detector_model/weights/best.pt  --source 'data/dataset/Rona_dataset_test/images/*.png' --img 960 --conf-thres 0.80 --iou-thres 0.6 --agnostic-nms

	
