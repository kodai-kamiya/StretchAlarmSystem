#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:00:50 2021

@author: kamiyakoudai
"""

import math
import time

import cv2
import gluoncv as gcv
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from gluoncv import model_zoo
from gluoncv.data.transforms.pose import (detector_to_alpha_pose,
                                          detector_to_simple_pose,
                                          heatmap_to_coord)
from gluoncv.utils import try_import_cv2
from PIL import Image

cv2 = try_import_cv2()

detector = model_zoo.get_model(
    'ssd_512_mobilenet1.0_voc',
    pretrained=True)
pose_net = model_zoo.get_model(
    'simple_pose_resnet18_v1b',
    # 'simple_pose_resnet50_v1d',
    # 'simple_pose_resnet101_v1b',
    # 'simple_pose_resnet152_v1b',
    # 'mobile_pose_mobilenetv2_1.0',
    # 'mobile_pose_mobilenetv3_small',
    # 'mobile_pose_mobilenetv3_large',
    # 'alpha_pose_resnet101_v1b_coco',
    pretrained=True)

detector.reset_class(["person"],
                     reuse_weights=['person'])

detector.hybridize()
pose_net.hybridize()

anglelist = []

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("stretch_01.mov")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('result_05.mp4', fmt, frame_rate, size)

time.sleep(1)

# frame_count = 344
for i in range(frame_count):

    ret, frame = cap.read()
    if ret == False :#フレームが読み込めなかったら出る
            break
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    # frame = cv2.imread('./01.jpg')
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    

    x, img = gcv.data.transforms.presets.ssd.transform_test(
        frame, short=512, max_size=700
    )
    class_IDs, scores, bounding_boxes = detector(x)

    # pose_input, upscale_bbox = \
    #     detector_to_alpha_pose(img, class_IDs, scores, bounding_boxes)
    pose_input, upscale_bbox = \
        detector_to_simple_pose(img, class_IDs, scores, bounding_boxes)

    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

    pose_img = gcv.utils.viz.cv_plot_keypoints(img,
                                               pred_coords,
                                               confidence,
                                               class_IDs,
                                               bounding_boxes,
                                               scores,
                                               box_thresh=0.5,
                                               keypoint_thresh=0.0)
    
    writer.write(frame)


writer.release()
cap.release()
cv2.destroyAllWindows()
