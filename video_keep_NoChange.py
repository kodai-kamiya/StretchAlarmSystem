#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:59:24 2021

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
writer = cv2.VideoWriter('result_nochange.mp4', fmt, frame_rate, size)

time.sleep(1)

print(frame_count)

# frame_count = 344
for i in range(frame_count):

    ret, frame = cap.read()
    
    writer.write(frame)

writer.release()
cap.release()
cv2.destroyAllWindows()
