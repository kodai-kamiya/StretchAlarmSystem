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

anglelist_rightarm = []
anglelist_leftarm = []
anglelist_rightleg = []
anglelist_leftleg = []
anglelist_rightelbow = []
anglelist_leftelbow = []


# filepath = "motion2.MOV"

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("motion04.MOV")
time.sleep(1)

# print(cap.get(cv2.CAP_PROP_FPS))
start = time.time()
result = 0 
count = 0

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
                                               # keypoint_thresh=0.2)
                                               keypoint_thresh=0)
    
    pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
    
    cv2.imshow('pose_img', pose_img)
    
    # create vectors
    
    ba = (pred_coords[0][7] - pred_coords[0][5]).asnumpy()
    bc = (pred_coords[0][11] - pred_coords[0][5]).asnumpy()
    angle_rightarm = angle_betweeen_two_vectors(ba, bc)
    anglelist_rightarm.append(angle_rightarm)
    
    de = (pred_coords[0][8] - pred_coords[0][6]).asnumpy()
    df = (pred_coords[0][12] - pred_coords[0][6]).asnumpy()
    angle_lefttarm = angle_betweeen_two_vectors(de, df)
    anglelist_leftarm.append(angle_lefttarm)
    
    gh = (pred_coords[0][5] - pred_coords[0][11]).asnumpy()
    gi = (pred_coords[0][13] - pred_coords[0][11]).asnumpy()
    angle_rightleg = angle_betweeen_two_vectors(gh, gi)
    anglelist_rightleg.append(angle_rightleg)
    
    jk = (pred_coords[0][6] - pred_coords[0][12]).asnumpy()
    jl = (pred_coords[0][14] - pred_coords[0][12]).asnumpy()
    angle_leftleg = angle_betweeen_two_vectors(jk, jl)
    anglelist_leftleg.append(angle_leftleg)
    
    mn = (pred_coords[0][9] - pred_coords[0][7]).asnumpy()
    mo = (pred_coords[0][5] - pred_coords[0][7]).asnumpy()
    angle_rightelbow = angle_betweeen_two_vectors(mn, mo)
    anglelist_rightelbow.append(angle_rightelbow)
    
    pq = (pred_coords[0][6] - pred_coords[0][8]).asnumpy()
    pr = (pred_coords[0][10] - pred_coords[0][8]).asnumpy()
    angle_leftelbow = angle_betweeen_two_vectors(pq, pr)
    anglelist_leftelbow.append(angle_leftelbow)
            
        
    if 160 <= angle_rightarm <= 180\
        and 160 <= angle_lefttarm <= 180\
            and 130 <= angle_rightelbow <= 180\
                and 130 <= angle_leftelbow <= 180\
                    and 130 <= angle_rightleg <= 180\
                        and 130 <= angle_leftleg <= 180 :
                            print("good")
    
    else :
        print("bad")
        count += 1
    
    result = int(time.time() - start)/10
    print(f"timer：{result}")
    
if count > cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4 : 
    print("あなたの姿勢は間違っています。")
    
else :
    print("あなたの姿勢は正しいです。")
          
cap.release()
cv2.destroyAllWindows()

# fig = plt.figure("histogram")

# ax1 = fig.add_subplot(2, 3, 1)
# ax1.set_title("Right Arm")
# ax2 = fig.add_subplot(2, 3, 2)
# ax2.set_title("left Arm")
# ax3 = fig.add_subplot(2, 3, 3)
# ax3.set_title("Right Leg")
# ax4 = fig.add_subplot(2, 3, 4)
# ax4.set_title("Left Leg")
# ax5 = fig.add_subplot(2, 3, 5)
# ax5.set_title("Right Elbow")
# ax6 = fig.add_subplot(2, 3, 6)
# ax6.set_title("Left Elbow")

bins = np.arange(0, 180+1, 5)

# ax1.hist(anglelist_rightarm, bins)
# ax2.hist(anglelist_lefttarm, bins)
# ax3.hist(anglelist_rightleg, bins)
# ax4.hist(anglelist_leftleg, bins)
# ax5.hist(anglelist_rightelbow, bins)
# ax6.hist(anglelist_leftelbow, bins)

plt.figure()
plt.xticks(np.arange(0, 180+1, 10))
plt.hist(anglelist_rightarm, bins)

plt.figure()
plt.xticks(np.arange(0, 180+1, 10))
plt.hist(anglelist_leftarm, bins)

plt.figure()
plt.xticks(np.arange(0, 180+1, 10))
plt.hist(anglelist_rightleg, bins)

plt.figure()
plt.xticks(np.arange(0, 180+1, 10))
plt.hist(anglelist_leftleg, bins)

plt.figure()
plt.xticks(np.arange(0, 180+1, 10))
plt.hist(anglelist_rightelbow, bins)

plt.figure()
plt.xticks(np.arange(0, 180+1, 10))
plt.hist(anglelist_leftelbow, bins)

plt.tight_layout()
plt.show()

angles = np.stack((anglelist_rightarm, anglelist_leftarm,
                   anglelist_rightelbow, anglelist_leftelbow, anglelist_rightleg, anglelist_leftleg))

np.savetxt('data.txt', angles.T, fmt='%.5e')
