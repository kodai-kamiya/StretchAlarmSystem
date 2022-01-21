import dash
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
# from dash.html.A import A
# from dash.html.B import B  # , Input, Output, State
from dash_extensions.enrich import DashProxy, MultiplexerTransform, Input, Output, State
# from datetime import date, time
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from flask import Flask, Response
import cv2
import gluoncv as gcv
import gluoncv
# from gluoncv import data
from gluoncv.data.transforms import pose
# import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from gluoncv import model_zoo
from gluoncv.data.transforms.pose import (detector_to_alpha_pose,
                                          detector_to_simple_pose,
                                          heatmap_to_coord)
from gluoncv.utils import try_import_cv2
from PIL import Image
# import dash_dangerously_set_inner_html
# from datetime import datetime as dt
import datetime as dt

# 2つのベクトル間の角度を求める関数


def angle_betweeen_two_vectors(v1: np.ndarray, v2: np.ndarray):
    cos_theta = np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    size_of_angle = np.rad2deg(theta)
    return size_of_angle

# 色々やるクラス


Fl = False


class VideoCamera(object):

    def __init__(self):
        # 必要なメンバ変数を定義

        # ウェブカメラを使用
        self.filename = 0
        # 動画ファイルを使用
        # self.filename = 'mov2.mp4'
        self.video = cv2.VideoCapture(self.filename)
        # 姿勢推定を行うかどうかを決めるフラグ
        self.esti_flag = False
        # 姿勢推定を行ったフレーム数
        self.esti_frames = 0
        # 姿勢推定を行ったフレームのうちgoodとしたフレーム数
        self.good_esti_frames = [0, 0, 0, 0, 0, 0]

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

    # カメラを止めるメソッド
    def stop_camera(self):
        # print('camera stopped')
        self.video.release()

    # カメラを起動するメソッド
    def start_camera(self):
        # print('camera started')
        self.video.open(self.filename)

    # 1フレームを取得するメソッド(フラグにより姿勢推定するかどうか変化)
    def get_frame(self):
        while(True):
            success, image = self.video.read()
            if success:
                break
            else:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, image = self.video.read()

        if self.esti_flag:
            pose_img = self.pose_estimation(image)
            flipped_pose_img = cv2.flip(pose_img, 1)
            ret, jpeg = cv2.imencode('.jpg', flipped_pose_img)

        else:
            flipped_image = cv2.flip(image, 1)
            ret, jpeg = cv2.imencode('.jpg', flipped_image)

        return jpeg.tobytes()

    # 姿勢推定を行いgood/badを判定するメソッド
    def pose_estimation(self, frame):
        old = frame
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

        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)

        self.esti_frames += 1

        ba = (pred_coords[0][7] - pred_coords[0][5]).asnumpy()
        bc = (pred_coords[0][11] - pred_coords[0][5]).asnumpy()
        angle_rightarm = angle_betweeen_two_vectors(ba, bc)
        if 160 <= angle_rightarm <= 180:
            print('rarm good')
            self.good_esti_frames[0] += 1

        ba = (pred_coords[0][8] - pred_coords[0][6]).asnumpy()
        bc = (pred_coords[0][12] - pred_coords[0][6]).asnumpy()
        angle_lefttarm = angle_betweeen_two_vectors(ba, bc)
        if 160 <= angle_lefttarm <= 180:
            print('larm good')
            self.good_esti_frames[1] += 1

        ba = (pred_coords[0][5] - pred_coords[0][11]).asnumpy()
        bc = (pred_coords[0][13] - pred_coords[0][11]).asnumpy()
        angle_rightleg = angle_betweeen_two_vectors(ba, bc)
        if 130 <= angle_rightleg <= 180:
            print('rleg good')
            self.good_esti_frames[2] += 1

        ba = (pred_coords[0][6] - pred_coords[0][12]).asnumpy()
        bc = (pred_coords[0][14] - pred_coords[0][12]).asnumpy()
        angle_leftleg = angle_betweeen_two_vectors(ba, bc)
        if 130 <= angle_leftleg <= 180:
            print('lleg good')
            self.good_esti_frames[3] += 1

        ba = (pred_coords[0][9] - pred_coords[0][7]).asnumpy()
        bc = (pred_coords[0][5] - pred_coords[0][7]).asnumpy()
        angle_rightelbow = angle_betweeen_two_vectors(ba, bc)
        if 130 <= angle_rightelbow <= 180:
            print('relbow good')
            self.good_esti_frames[4] += 1

        ba = (pred_coords[0][6] - pred_coords[0][8]).asnumpy()
        bc = (pred_coords[0][10] - pred_coords[0][8]).asnumpy()
        angle_leftelbow = angle_betweeen_two_vectors(ba, bc)
        if 130 <= angle_leftelbow <= 180:
            print('lelbow good')
            self.good_esti_frames[5] += 1

        return pose_img
        # return old

    # 姿勢推定を開始するメソッド
    def start_esti(self):
        if not self.esti_flag:
            self.esti_flag = True

    # 姿勢推定を停止するメソッド
    def stop_esti(self):
        if self.esti_flag:
            self.esti_flag = False

    #  good/bad判定に用いる変数を0にクリアするメソッド
    def clear_esti_variable(self):
        self.esti_frames = 0
        self.good_esti_frames = [0, 0, 0, 0, 0, 0]

    # 姿勢推定の結果がgoodかどうかを判定するメソッド
    def is_good_judgement(self):
        # global Fl
        # if Fl is False:
        #     return False
        # else:
        #     return True
        print(self.good_esti_frames)
        print(self.esti_frames)
        if self.esti_frames == 0:
            return False
        else:
            for i in range(0, 6):
                if (self.good_esti_frames[i]/self.esti_frames)*100 < 60:
                    return False
            return True


server = Flask(__name__)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@server.route('/video_feed')
def video_feed():
    return Response(gen(camera_object),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# app = dash.Dash(__name__, server=server,
#                 external_stylesheets=[dbc.themes.BOOTSTRAP])


# 便利な設定
app = DashProxy(__name__, server=server,
                prevent_initial_callbacks=True,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                # meta_tags=[{"name": "viewport",
                #             "content": "width=device-width, initial-scale=1"}],
                transforms=[MultiplexerTransform(proxy_location=None)]
                )

# 時間を入力プルダウンに必要な変数を定義
hour = list(range(0, 24))
hour_options = []
for year in hour:
    hour_options.append({'label': str(year), 'value': year})

minute = list(range(0, 60))
minute_options = []
for year in minute:
    minute_options.append({'label': str(year), 'value': year})

# 必要なグローバル変数を定義

# アラームの日付を保持する変数
alarm_date = None
# VideoCameraオブジェクトを保持する変数
camera_object = None
# ストレッチ開始時刻を保持する変数
stretch_start_time = None


# レイアウト用の変数を定義
# show_text =\
#     html.Div('Stretching Alarm Clock', id='r_title')

# お手本画像
otehon_image =\
    html.Div(
        children=[
            html.Img(src=app.get_asset_url('stretching.jpg'),
                     width="auto", height="100%")
        ], id='image_container'
    )
# 取得したビデオ画像
video_image =\
    html.Div(
        children=[html.Img(src="/video_feed", alt="video", width="auto", height="100%")], id='video_area'
    )

# アラームをセットするボタン
set_button = dbc.Button("Set", color="primary",
                        className="d-grid gap-2 col-6 mx-auto", id='set_button')
# アラームを止めるボタン
stop_button = dbc.Button("Stop",
                         color="danger", className="d-grid gap-2 col-6 mx-auto", id='stop_button', href='/stretch')

stop_button2 = dbc.Button("Stop",
                          color="danger", className="d-grid gap-2 col-6 mx-auto", id='stop_button2')
# アラームを中止するボタン
cancel_button = dbc.Button("Cancel",
                           color="danger", className="d-grid gap-2 col-6 mx-auto", id='cancel_button')

# test_button = dbc.Button("test",
#                          color="danger", className="me-1", id='test_button')

# ルートに戻るボタン
back_button = dbc.Button("back",
                         color="danger", className="d-grid gap-2 col-6 mx-auto", id='back_button', href='/')

# ルート画面のメイン領域(長いため分離)
r_main_area =\
    html.Div(
        children=[
            html.Div(children=[html.Div('Please Set Alarm Clock',
                                        id='r_one_comment_area')], id='r_one_comment_area_container'),
            html.Div(
                children=[
                    dcc.Interval(
                        id='r_interval_component',
                        interval=1*1000,
                        n_intervals=0
                    ),
                    html.Div('Date', id='r_date_msg_area'),
                    html.Div(
                        children=[

                            dcc.DatePickerSingle(
                                id='r_my-date-picker-single',
                                min_date_allowed=dt.date(2021, 1, 1),
                                max_date_allowed=dt.date(2022, 12, 31),
                                # initial_visible_month=dt.date(2017, 8, 5),
                                # date=dt.date(2021, 8, 25),
                                date=dt.date.today(),
                                display_format='Y/M/D'),

                        ], id='r_date_input_area'),
                    html.Div('Time', id='r_time_msg_area'),
                    html.Div(children=[
                        dcc.Dropdown(
                            id='r_hour-picker', options=hour_options, value='0'),
                        html.Div(':', id='r_colon_area'),
                        dcc.Dropdown(
                            id='r_minute-picker', options=minute_options, value='0')
                    ], id='r_hour_and_minute_input_area'),
                    html.Div(children=[
                        set_button,
                    ], id='r_button_area')

                ], id='r_alarm_input_area_container'),
            html.Div(children=[
                html.Div(children=None, id='r_msg1'),
                html.Div(children=None, id='r_msg2'),
            ], id='r_msg_area'),
        ], id='r_alarm_area_container')

# ルート(/)のレイアウト
r = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div('Stretching Alarm Clock', id='r_title'),
                    width=12,
                    className='bg-light',
                    id='r_title_area'
                ),
            ],
            align='center', style={"height": "10vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    r_main_area,
                    width={"size": 6, "offset": 3},
                    className='bg-secondary',
                    id='r_alarm_area'
                ),
            ],
            align='center', style={"height": "80vh"}
        ),
        dbc.Row(None,
                align='center', style={"height": "10vh"}
                ),
    ],
    fluid=True
)

# ストレッチ画面のメイン領域
s_main_area =\
    [
        # html.Div(id='hidden-div', style={'display': 'none'}),
        html.Div(children=None, id='s_msg1'),
        html.Div(children=None, id='s_msg2'),
        html.Div(children=None, id='s_back_button_area')]

# ストレッチ(/stretch)画面のレイアウト
s = dbc.Container(
    [
        dcc.Interval(
            id='s_interval_component',
            interval=1*1000,  # in milliseconds
            n_intervals=0
        ),
        dbc.Row(
            [
                dbc.Col(
                    children=s_main_area,
                    width=12,
                    className='bg-light',
                    id='s_title_area',
                    style={'height': '100%'}
                ),
            ],
            align='center', style={"height": "20vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    otehon_image,
                    width=6,
                    className='bg-primary',
                    id='image_area'
                ),
                dbc.Col(
                    video_image,
                    width=6,
                    className='bg-danger',
                    id='video_container'
                ),
            ],
            align='center', style={"height": "80vh"}
        ),
    ],
    fluid=True
)

# アプリのレイアウト
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

isRinging = False


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    global camera_object
    global stretch_start_time
    stretch_start_time = None
    if camera_object is not None:
        camera_object.stop_esti()
        camera_object.clear_esti_variable()

    if pathname == '/':
        if camera_object is not None:
            camera_object.stop_camera()
        return r
    elif pathname == '/stretch':
        if camera_object is None:
            camera_object = VideoCamera()
        else:
            camera_object.start_camera()
        return s
    else:
        if camera_object is not None:
            camera_object.stop_camera()
        return '404'


@app.callback(Output('r_button_area', 'children'),
              Output('r_msg2', 'children'),
              Input('set_button', 'n_clicks'),
              State('r_my-date-picker-single', 'date'),
              State('r_hour-picker', 'value'),
              State('r_minute-picker', 'value'), prevent_initial_call=True)
def set_button_pushed(set_n, date_value, hour_value, minute_value):
    if (set_n is None) or (set_n == 0):
        raise PreventUpdate
    else:
        time_str = date_value + "-"+str(hour_value)+"-"+str(minute_value)+"-0"
        set_time = dt.datetime.strptime(time_str, '%Y-%m-%d-%H-%M-%S')
        current_time = dt.datetime.now()
        if set_time > current_time:
            global alarm_date
            alarm_date = set_time
            return cancel_button, 'You set alarm clock'
        else:
            return dash.no_update, 'Please set again'


@ app.callback(Output('r_button_area', 'children'),
               Output('r_msg2', 'children'),
               Input('stop_button', 'n_clicks'),
               prevent_initial_call=True)
def stop_button_pushed(stop_button):
    if (stop_button is None) or (stop_button == 0):
        raise PreventUpdate
    else:
        global alarm_date
        # print(alarm_date)
        alarm_date = None
        # print(alarm_date)
        return set_button, None


@ app.callback(Output('r_button_area', 'children'),
               Output('r_msg2', 'children'),
               Input('cancel_button', 'n_clicks'),
               prevent_initial_call=True)
def cancel_button_pushed(cancel_n):
    if (cancel_n is None) or (cancel_n == 0):
        raise PreventUpdate
    else:
        global alarm_date
        print(alarm_date)
        alarm_date = None
        print(alarm_date)
        return set_button, 'You canceled alarm clock'


@ app.callback(Output('r_button_area', 'children'),
               Output('r_msg1', 'children'),
               Input('r_interval_component', 'n_intervals'))
def check_alarm(interval_n):
    today = dt.datetime.now()
    global alarm_date
    if alarm_date is None:
        return dash.no_update, '{0}/{1}/{2} {3}:{4}:{5}'.format(today.year, str(today.month).zfill(2), str(today.day).zfill(2), str(today.hour).zfill(2), str(today.minute).zfill(2), str(today.second).zfill(2))
    else:
        if alarm_date > today:
            date_diff = alarm_date - today
            hours, tminute = divmod(date_diff.seconds, 3600)
            minutes, seconds = divmod(tminute, 60)
            return dash.no_update, '{0}days {1}:{2}:{3}'.format(date_diff.days, str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2))
        else:
            return stop_button, html.Audio(html.Source(src=f"https://www.ne.jp/asahi/music/myuu/wave/loop1.wav", type="audio/wav"), autoPlay=True, loop=True, preload='auto')


@ app.callback(Output('s_back_button_area', 'children'),
               Input('stop_button2', 'n_clicks'))
def a(n):
    if (n is None) or (n == 0):
        raise PreventUpdate
    else:
        global isRinging
        isRinging = False
        return None


@ app.callback(Output('s_msg1', 'children'),
               Output('s_msg2', 'children'),
               Output('s_back_button_area', 'children'),
               Input('s_interval_component', 'n_intervals'))
def check_stretch(interval_n):
    global isRinging
    if isRinging:
        return dash.no_update, dash.no_update, dash.no_update
    else:
        global stretch_start_time
        if not stretch_start_time:
            stretch_start_time = dt.datetime.now()
        elapsed_time = (dt.datetime.now() - stretch_start_time).seconds
        if elapsed_time < 11:
            return 'Start in ' + str(10 - elapsed_time) + ' seconds', None, None
        elif 11 <= elapsed_time < 21:
            camera_object.start_esti()
            return 'Stretch Now!', str(20 - elapsed_time), None
        else:
            camera_object.stop_esti()
            if camera_object.is_good_judgement():
                return None, 'Clear!', back_button
            else:
                camera_object.clear_esti_variable()
                stretch_start_time = None
                isRinging = True
                # global Fl
                # Fl = True
                return 'You failed', html.Audio(html.Source(src=f"https://www.ne.jp/asahi/music/myuu/wave/loop1.wav", type="audio/wav"), autoPlay=True, loop=True, preload='auto'), stop_button2


if __name__ == '__main__':
    app.run_server(debug=True)
