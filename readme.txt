model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")

gedit /home/zb/.config/Ultralytics/settings.json


然后进入ultralytics 目录pip install -e .
(utralytics) zb@neu:/data_jiang/zb/Simon/mul_yolov11_detect/yolov11$ python train.py
#(utralytics) zb@neu:/data_jiang/zb/Simon/mul_yolov11_detect/yolov11$ python val.py
(utralytics) zb@neu:/data_jiang/zb/Simon/mul_yolov11_detect/yolov11$ python detect.py
(utralytics) zb@neu:/data_jiang/zb/Simon/mul_yolov11_detect/yolov11$ python Xdetect.py


