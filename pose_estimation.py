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


def angle_betweeen_two_vectors(v1: np.ndarray, v2: np.ndarray):
    cos_theta = np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    size_of_angle = np.rad2deg(theta)
    return size_of_angle


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

filepath = "filepath"

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(filepath)
time.sleep(1)

while(True):

    ret, frame = cap.read()

    if not ret:
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
                                               keypoint_thresh=0.2)

    cv2.imshow('pose_img', pose_img)

    # create vectors
    ba = (pred_coords[0][7] - pred_coords[0][5]).asnumpy()
    bc = (pred_coords[0][11] - pred_coords[0][5]).asnumpy()
    angle = angle_betweeen_two_vectors(ba, bc)
    anglelist.append(angle)
    print(angle)

print(anglelist)

cap.release()
cv2.destroyAllWindows()

plt.hist(anglelist, bins=36)
plt.show()
