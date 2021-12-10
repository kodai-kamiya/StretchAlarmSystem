import dash
from dash import Dash, dcc, html, Input, Output, State
from datetime import date, time
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from flask import Flask, Response
import cv2
import gluoncv as gcv
from gluoncv import data
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from gluoncv import model_zoo
from gluoncv.data.transforms.pose import (detector_to_alpha_pose,
                                          detector_to_simple_pose,
                                          heatmap_to_coord)
from gluoncv.utils import try_import_cv2
from PIL import Image
import dash_dangerously_set_inner_html
from datetime import datetime as dt
import datetime
import plotly.express as px
from jupyter_dash import JupyterDash


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
        # flipped_pose_img = cv2.flip(pose_img, 1)
        # ret, jpeg = cv2.imencode('.jpg', flipped_pose_img)

        # do not pose estimation
        flipped_image = cv2.flip(image, 1)
        ret, jpeg = cv2.imencode('.jpg', flipped_image)

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
hour = list(range(0, 24))
hour_options = []
for year in hour:
    hour_options.append({'label': str(year), 'value': year})

minute = list(range(0, 60))
minute_options = []
for year in minute:
    minute_options.append({'label': str(year), 'value': year})

alarm = ['']

external_stylesheets = [dbc.themes.BOOTSTRAP]


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


show_text =\
    html.Div('Stretching Alarm Clock', id='text_cell')

show_main =\
    html.Div(
        children=[
            html.Div(children=[html.Div('Please Set Alarm Clock',
                                        id='one_comment_area')], id='one_comment_area_container'),
            html.Div(
                children=[
                    html.Div('Date', id='date_msg_area'),
                    html.Div(
                        children=[

                            dcc.DatePickerSingle(
                                id='my-date-picker-single',
                                min_date_allowed=date(2021, 1, 1),
                                max_date_allowed=date(2022, 12, 31),
                                # initial_visible_month=date(2017, 8, 5),
                                # date=date(2021, 8, 25),
                                date=datetime.date.today(),
                                display_format='Y/M/D'),

                        ], id='date_input_area'),
                    html.Div('Time', id='time_msg_area'),
                    html.Div(children=[
                        dcc.Dropdown(
                            id='hour-picker', options=hour_options, value='0'),
                        html.Div(':', id='colon_area'),
                        dcc.Dropdown(
                            id='minute-picker', options=minute_options, value='0')
                    ], id='hour_and_minute_input_area'),
                    html.Div(children=[
                        dbc.Button("Set", outline=True,
                                   color="primary", className="me-1"),
                        dbc.Button("Stop", outline=True,
                                   color="danger", className="me-1"),
                    ], id='button_area')

                ], id='alarm_input_area_container'),
            html.Div('message area', id='msg_area'),
        ], id='alarm_area_container')

show_image =\
    html.Div(
        children=[
            html.Img(src=app.get_asset_url('Lenna.png'),
                     width="auto", height="100%")
        ], id='image_container'
    )

show_video =\
    html.Div(
        children=[
            html.Img(src="/video_feed", alt="video",
                     width="auto", height="100%")
        ], id='video_container'
    )


l2 = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    show_text,
                    width=12,
                    className='bg-light',
                    id='title_area'
                ),
            ],
            align='center', style={"height": "8vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    show_main,
                    width={"size": 6, "offset": 3},
                    className='bg-secondary',
                    id='alarm_area'
                ),
            ],
            align='center', style={"height": "47vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    show_image,
                    width=6,
                    className='bg-primary',
                    id='image_area'
                ),
                dbc.Col(
                    show_video,
                    width=6,
                    className='bg-danger',
                    id='video_area'
                ),
            ],
            align='center', style={"height": "45vh"}
        ),
    ],
    fluid=True
)

l1 = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div([

                        html.Button('Set', id='set-val', n_clicks=0),
                        html.Div(id='my-output'),
                        html.Div(id='live-update-text'),
                        html.Div(id='live-update-text2'),
                        html.Div(id='live-update-text3'),

                        dcc.Interval(
                            id='interval-component',
                            interval=1*1000,  # in milliseconds
                            n_intervals=0
                        )
                    ]),
                    # html.H1("タイトル"),
                    style={"background-color": "pink"}
                )
            ],
            style={"height": "30vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.DatePickerSingle(
                                        id='my-date-picker-single',
                                        min_date_allowed=date(2021, 1, 1),
                                        max_date_allowed=date(2022, 12, 31),
                                        # initial_visible_month=date(2017, 8, 5),
                                        # date=date(2021, 8, 25),
                                        date=datetime.date.today(),
                                        display_format='Y/M/D',)
                                ),
                                dbc.Col(dcc.Dropdown(
                                    id='hour-picker', options=hour_options, value='1')),
                                dbc.Col(dcc.Dropdown(
                                    id='minute-picker', options=minute_options, value='1')),

                            ]),

                    ]),
                    # html.P("アラーム機能"),
                    style={"height": "100%", "background-color": "red"}
                )
            ],
            style={"height": "35vh"}
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
            style={"height": "35vh"}
        )
    ],
    fluid=True
)

app.layout = l2


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    today = dt.now()
    if not alarm[0]:
        return [html.Span('今は{0}年{1}月{2}日{3}時{4}分{5}秒です'.format(today.year, today.month, today.day, today.hour, today.minute, today.second))]
    else:
        print(alarm[0])
        setted_time = dt.strptime(alarm[0], '%Y-%m-%d-%H-%M-%S')
        if setted_time > today:
            date_diff = setted_time - today
            hours, tminute = divmod(date_diff.seconds, 3600)
            minutes, seconds = divmod(tminute, 60)
            return [html.Span('アラームまであと{0}日{1}時間{2}分{3}秒です'.format(date_diff.days, hours, minutes, seconds)),
                    html.Button('stop', id='alarm-count-stop-val', n_clicks=0)]
        else:
            return [html.Span('アラームがなっています'), html.Button('stop', id='alarming-stop-val', n_clicks=0)]


@app.callback(Output('live-update-text2', 'children'),
              Input('alarm-count-stop-val', 'n_clicks'))
def alarm_cancel(n):
    if n > 0:
        print("alarm_cancel")
        alarm[0] = ''
        # return {'display': 'none'}
        return [html.Span('アラームをキャンセルしました')]


@app.callback(Output('live-update-text3', 'children'),
              Input('alarming-stop-val', 'n_clicks'))
def alarm_stop(n):
    if n > 0:
        print("alarm_stop")
        alarm[0] = ''
        return [html.Span('アラームを停止しました')]


@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input('set-val', 'n_clicks')],
    [State('my-date-picker-single', 'date'),
     State('hour-picker', 'value'),
     State('minute-picker', 'value')]
)
def update_output(n_clicks, date_value, hour_value, minute_value):
    if n_clicks > 0:
        time_str = date_value + "-"+str(hour_value)+"-"+str(minute_value)+"-0"

        setted_time = dt.strptime(time_str, '%Y-%m-%d-%H-%M-%S')
        if setted_time < dt.today():
            return '未来の時間を入力してください'
        else:
            alarm[0] = setted_time.strftime('%Y-%m-%d-%H-%M-%S')
            return '{0}年{1}月{2}日の{3}時{4}分にアラームをセットしました'.format(setted_time.year, setted_time.month, setted_time.day, setted_time.hour, setted_time.minute)


if __name__ == '__main__':
    app.run_server(debug=True)
