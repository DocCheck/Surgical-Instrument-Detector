#!/bin/bash                                                                                                         


cp -r ./Yolo_changes/Makefile ./yolov5/
cp -r ./Yolo_changes/Dockerfile ./yolov5/
cp  ./Yolo_changes/requirements.txt ./yolov5/
cp  ./Yolo_changes/requirements_cam_virtenv.txt ./yolov5/


cp ./Yolo_changes/rona.hyp.scratch.yaml ./yolov5/data/hyps/
cp ./Yolo_changes/dataloaders.py ./yolov5/utils/
cp ./Yolo_changes/augmentations.py ./yolov5/utils/
cp -r ./Yolo_changes/preprocessing ./yolov5/

mkdir -p ./yolov5/data/dataset/Rona_dataset_final_single_multi_fit/
cp -r ./Yolo_changes/dataset/data.yaml ./yolov5/data/dataset/Rona_dataset_final_single_multi_fit/
