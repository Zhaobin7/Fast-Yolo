import torch
from PIL import Image
import os
from ultralytics import YOLO
# 加载 YOLOv5 模型
model = YOLO("/home/zb/zb/Simon/mul_yolov11_detect/yolov11/runs/detect/defect/weights/best.pt")

model.predict(source='images_defect/train/images/',
              imgsz=640,
              project='runs/results',
              name='exp',
              save=True,
              # conf=0.2,
              # iou=0.7,
              # agnostic_nms=True,
              # visualize=True, # visualize model features maps
              # line_width=2, # line width of the bounding boxes
              # show_conf=False, # do not show prediction confidence
              # show_labels=False, # do not show prediction labels
              # save_txt=True, # save results as .txt file
              # save_crop=True, # save cropped images with results
            )
'''

# 输入目录
input_directory = './images_defect/train/images/'  # 图片所在的目录

# 遍历目录中的图片文件
for filename in os.listdir(input_directory):
    print("XXXXXXXXXXXXXXXXXXXXXXX111111111111111111111111111111111111111")
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 过滤图片文件
        image_path = os.path.join(input_directory, filename)

        # 加载图片
        img = Image.open(image_path)

        # 使用模型进行预测
        #results = model(img)
        results = model.predict(input_directory, imgsz=640, batch=16, conf=0.25, iou=0.6, device="1")
        # 获取预测结果（通常是包括边界框、类别标签等）
        predicted_image = results.render()[0]  # 渲染后的图像，带有预测结果
        labels = results.names  # 类别标签
        pred_labels = results.xywh[0][:, -1].tolist()  # 预测的标签索引
        # 打印或保存文件名和预测结果
        print(f"预测结果 for {filename}:")
        for label_idx in pred_labels:
            print("3333333333333333XXXXXXXXXXXXXXXXXX111111111111111111111111111111111111111")
            print(f"  识别到物体: {labels[int(label_idx)]}")

        # 保存预测结果图像到指定目录
        output_directory = '/home/zb/zb/Simon/mul_yolov11_detect/yolov11/runs/results'
        if not os.path.exists(output_directory):
            print("1111111111111111111111111111111111111111111111111111")
            os.makedirs(output_directory)

        save_path = os.path.join(output_directory, filename)
        Image.fromarray(predicted_image).save(save_path)
        print(f"预测结果保存到: {save_path}")
'''
print("所有图片已处理完毕！")
