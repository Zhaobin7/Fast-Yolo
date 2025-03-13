'''
Author: wy 1955416359@qq.com
Date: 2024-12-21 13:04:35
LastEditors: wy 1955416359@qq.com
LastEditTime: 2024-12-24 22:34:47
FilePath: /wy/VOCdevkitPCB/dash/train.py
Description: 

'''
from ultralytics import YOLO
import os

"""
import zipfile
with zipfile.ZipFile(f"./green_dataset.v1i.yolov11.zip","r") as zip_ref:
    zip_ref.extractall("data")
"""

#/home/zb/zb/Simon/mul_yolov11_detect/yolov11/cfg/yolo11s_scdown_involution.yaml
# Load the model.
# model = YOLO(f'./yolov8s.pt')

#model = YOLO("./cfg/11/yolo11-C3k2-AdditiveBlock-CGLU.yaml").load("./yolo11s.pt")  # build from YAML and transfer weights
model = YOLO("./cfg/yolo11s_smalldet.yaml").load("./yolo11s.pt")  # build from YAML and transfer weights
#model = YOLO("./cfg/yolo11s_scdown_involution.yaml").load("./yolo11s.pt")  # build from YAML and transfer weights
# Training.
results = model.train(
   data=os.path.abspath(f"./data/data11_defect.yaml"),
   imgsz=800,
   epochs=500,
   batch=16,
   name='yolo11s_smalldet_300e',
   device=[0],
   nwd_loss = False
)

val_result = model.val()
