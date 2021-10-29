from gluoncv import model_zoo
from gluoncv.data.transforms.pose import (
    detector_to_simple_pose,
    detector_to_alpha_pose,
    heatmap_to_coord
)
import mxnet as mx
import time
import gluoncv as gcv
from gluoncv.utils import try_import_cv2

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


cap = cv2.VideoCapture(0)
time.sleep(1)

while(True):

    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

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


cap.release()
cv2.destroyAllWindows()
