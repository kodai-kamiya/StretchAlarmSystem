import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from flask import Flask, Response
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


class VideoCamera(object):
    def __init__(self):
        # capture from video file
        # filename = "filepath"
        # self.video = cv2.VideoCapture(filename)

        # capture from webcam
        self.video = cv2.VideoCapture(0)

        self.detector = model_zoo.get_model(
            'ssd_512_mobilenet1.0_voc',
            pretrained=True)
        self.pose_net = model_zoo.get_model(
            'simple_pose_resnet18_v1b',
            # 'simple_pose_resnet50_v1d',
            # 'simple_pose_resnet101_v1b',
            # 'simple_pose_resnet152_v1b',
            # 'mobile_pose_mobilenetv2_1.0',
            # 'mobile_pose_mobilenetv3_small',
            # 'mobile_pose_mobilenetv3_large',
            # 'alpha_pose_resnet101_v1b_coco',
            pretrained=True)
        self.detector.reset_class(["person"],
                                  reuse_weights=['person'])
        self.detector.hybridize()
        self.pose_net.hybridize()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while(True):
            success, image = self.video.read()
            if success:
                break
            else:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, image = self.video.read()

        # do pose estimation
        # pose_img = self.pose_estimation(image)
        # ret, jpeg = cv2.imencode('.jpg', pose_img)

        # do not pose estimation
        jpeg = image

        # flip image horizontal
        ret, jpeg = cv2.imencode('.jpg', cv2.flip(jpeg, 1))

        # don't flip image
        # ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()

    def pose_estimation(self, frame):
        frame = mx.nd.array(cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB)).astype('uint8')

        x, img = gcv.data.transforms.presets.ssd.transform_test(
            frame, short=512, max_size=700
        )
        class_IDs, scores, bounding_boxes = self.detector(x)

        # pose_input, upscale_bbox = \
        #     detector_to_alpha_pose(img, class_IDs, scores, bounding_boxes)
        pose_input, upscale_bbox = \
            detector_to_simple_pose(img, class_IDs, scores, bounding_boxes)

        predicted_heatmap = self.pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(
            predicted_heatmap, upscale_bbox)

        pose_img = gcv.utils.viz.cv_plot_keypoints(img,
                                                   pred_coords,
                                                   confidence,
                                                   class_IDs,
                                                   bounding_boxes,
                                                   scores,
                                                   box_thresh=0.5,
                                                   keypoint_thresh=0.2)
        return pose_img


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


server = Flask(__name__)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[dbc.themes.BOOTSTRAP])


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# app.layout = html.Div([
#     html.H1("Webcam Test"),
#     html.Img(src="/video_feed")
# ])

# html layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1("タイトル"),
                    style={"background-color": "pink"}
                )
            ],
            style={"height": "10vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.P("アラーム機能"),
                    style={"height": "100%", "background-color": "red"}
                )
            ],
            style={"height": "45vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    # html.P("お手本表示"),
                    html.Img(src=app.get_asset_url('Lenna.png'),
                             width="auto", height="100%"),
                    width=6,
                    style={"text-align": "center", "height": "100%",
                           "background-color": "blue"}
                ),
                dbc.Col(children=[
                    # html.P("カメラ画像表示"),
                    html.Img(src="/video_feed", alt="video", width="auto", height="100%")],
                    width=6,
                    style={"display": "block",
                           "text-align": "center", "height": "100%",
                           "background-color": "cyan"}
                )
            ],
            style={"height": "45vh"}
        )
    ],
    fluid=True
)


if __name__ == '__main__':
    app.run_server(debug=True)
