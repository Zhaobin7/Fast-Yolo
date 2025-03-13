'''
Author: wy 1955416359@qq.com
Date: 2024-12-19 14:07:14
LastEditors: wy 1955416359@qq.com
LastEditTime: 2024-12-21 14:48:50
FilePath: /wy/VOCdevkitPCB/dash/val.py
Description: 

'''
from ultralytics import YOLO

# Load a model
model = YOLO("/data_jiang/zb/Simon/ultralytics/runs/detect/yolo11s_smalldet_300e10/weights/best.pt")

# Validate the model
# metrics = model.val(data="./data/data.yaml")
metrics = model.val(data="./data/data19_workpiece.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="1")
# print(metrics.box.map)  # map50-95
