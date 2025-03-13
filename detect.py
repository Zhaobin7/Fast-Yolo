import os
import torch
from PIL import Image
from ultralytics import YOLO

# Load a model
model = YOLO("/home/zb/zb/Simon/mul_yolov11_detect/yolov11/runs/detect/defect/weights/best.pt")


# 输入目录
input_directory = './images_defect/train/images/'  # 图片所在的目录
output_directory = './runs/results'         # 保存预测结果图像到指定目录

# ç¡®ä¿èŸåºç®åœå­åš
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


results = model.predict(input_directory, imgsz=640, batch=16, conf=0.25, iou=0.6, device="1")
        
# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    #r.show()
    #print("11111111111111111111",os.path.split(results.path))
    # Save results to disk
    #r.save(output_directory)
    filename = f"results{i}.jpg"
    images_folder = os.path.join(output_directory, filename)
    #print("1111111111111111111111",images_folder)
    r.save(images_folder)
