import dash
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
from dash.html.A import A
from dash.html.B import B  # , Input, Output, State
from dash_extensions.enrich import DashProxy, MultiplexerTransform, Input, Output, State
from datetime import date, time
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from flask import Flask, Response
import cv2
import gluoncv as gcv
from gluoncv import data
from gluoncv.data.transforms import pose
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from gluoncv import model_zoo
from gluoncv.data.transforms.pose import (detector_to_alpha_pose,
                                          detector_to_simple_pose,
                                          heatmap_to_coord)
from gluoncv.utils import try_import_cv2
from PIL import Image
# import dash_dangerously_set_inner_html
from datetime import datetime as dt
import datetime

# from a import example
# import plotly.express as px
# from jupyter_dash import JupyterDash


# filename = "mov1.mp4"

def angle_betweeen_two_vectors(v1: np.ndarray, v2: np.ndarray):
    cos_theta = np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    size_of_angle = np.rad2deg(theta)
    return size_of_angle


class VideoCamera(object):
    def __init__(self):
        print('instanced')
        self.filename = "mov2.mp4"
        # capture from video file
        self.video = cv2.VideoCapture(self.filename)

        self.esti_flag = False

        self.all_frames = 0
        self.good_counts = 0

        # capture from webcam
        # self.video = cv2.VideoCapture(0)

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
        print('I\'ll die')
        self.video.release()
        # del self.video
        # cv2.destroyAllWindows()

    def stop_camera(self):
        print('camera stopped')
        self.video.release()

    def start_camera(self):
        print('camera restart')
        self.video.open(self.filename)

    def get_frame(self):
        while(True):
            success, image = self.video.read()
            if success:
                break
            else:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, image = self.video.read()

        # do pose estimation
        if self.esti_flag:
            pose_img = self.pose_estimation(image)
            flipped_pose_img = cv2.flip(pose_img, 1)
            ret, jpeg = cv2.imencode('.jpg', flipped_pose_img)
        # do not pose estimation
        else:
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

        self.all_frames += 1

        ba = (pred_coords[0][7] - pred_coords[0][5]).asnumpy()
        bc = (pred_coords[0][11] - pred_coords[0][5]).asnumpy()
        angle_rightarm = angle_betweeen_two_vectors(ba, bc)
        if not (160 <= angle_rightarm <= 180):
            return pose_img

        ba = (pred_coords[0][8] - pred_coords[0][6]).asnumpy()
        bc = (pred_coords[0][12] - pred_coords[0][6]).asnumpy()
        angle_lefttarm = angle_betweeen_two_vectors(ba, bc)
        if not (160 <= angle_lefttarm <= 180):
            return pose_img

        ba = (pred_coords[0][5] - pred_coords[0][11]).asnumpy()
        bc = (pred_coords[0][13] - pred_coords[0][11]).asnumpy()
        angle_rightleg = angle_betweeen_two_vectors(ba, bc)
        if not (130 <= angle_rightleg <= 180):
            return pose_img

        ba = (pred_coords[0][6] - pred_coords[0][12]).asnumpy()
        bc = (pred_coords[0][14] - pred_coords[0][12]).asnumpy()
        angle_leftleg = angle_betweeen_two_vectors(ba, bc)
        if not (130 <= angle_leftleg <= 180):
            return pose_img

        ba = (pred_coords[0][9] - pred_coords[0][7]).asnumpy()
        bc = (pred_coords[0][5] - pred_coords[0][7]).asnumpy()
        angle_rightelbow = angle_betweeen_two_vectors(ba, bc)
        if not (130 <= angle_rightelbow <= 180):
            return pose_img

        ba = (pred_coords[0][6] - pred_coords[0][8]).asnumpy()
        bc = (pred_coords[0][10] - pred_coords[0][8]).asnumpy()
        angle_leftelbow = angle_betweeen_two_vectors(ba, bc)
        if not (130 <= angle_leftelbow <= 180):
            return pose_img

        self.good_counts += 1

        return pose_img

    def get_judge(self):
        # print(self.all_frames)
        # print(self.good_counts)

        if self.all_frames == 0:
            return False
        else:
            good_percent = (self.good_counts / self.all_frames) * 100
            if(good_percent > 60):
                return True
            else:
                return False


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


server = Flask(__name__)
# app = dash.Dash(__name__, server=server,
#                 external_stylesheets=[dbc.themes.BOOTSTRAP])


app = DashProxy(__name__, server=server,
                prevent_initial_callbacks=True,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                # meta_tags=[{"name": "viewport",
                #             "content": "width=device-width, initial-scale=1"}],
                transforms=[MultiplexerTransform(proxy_location=None)]
                )

hour = list(range(0, 24))
hour_options = []
for year in hour:
    hour_options.append({'label': str(year), 'value': year})

minute = list(range(0, 60))
minute_options = []
for year in minute:
    minute_options.append({'label': str(year), 'value': year})

# alarm = ['']
# alarm_dates = []
alarm_date = None

is_alarm_stopped = False
# x = VideoCamera()

# camera_objects = []
camera_object = None


@server.route('/video_feed')
def video_feed():
    return Response(gen(camera_object),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(gen(camera_objects[0]),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')


show_text =\
    html.Div('Stretching Alarm Clock', id='text_cell')

set_button = dbc.Button("Set", outline=True, color="primary",
                        className="me-1", id='set_button')

# stop_button = dbc.Button("Stop", outline=True,
#                          color="danger", className="me-1", id='stop_button')
stop_button = dbc.Button("Stop", outline=True,
                         color="danger", className="me-1", id='stop_button', href='/stretch')

# cancel_button = dbc.Button("Cancel", outline=True,
#                            color="danger", className="me-1", id='cancel_button', href='/stretch')
cancel_button = dbc.Button("Cancel", outline=True,
                           color="danger", className="me-1", id='cancel_button')

test_button = dbc.Button("test", outline=True,
                         color="danger", className="me-1", id='test_button')

# back_button = [dbc.Button("back", outline=True,
#                           color="danger", className="me-1", id='back_button', href='/'),
#                dbc.Button("start", outline=True,
#                           color="primary", className="me-1", id='video_start_button'),
#                dbc.Button("stop", outline=True,
#                color="danger", className="me-1", id='video_stop_button')]

show_main =\
    html.Div(
        children=[
            html.Div(children=[html.Div('Please Set Alarm Clock',
                                        id='one_comment_area')], id='one_comment_area_container'),
            html.Div(
                children=[
                    dcc.Interval(
                        id='interval_component',
                        interval=1*1000,  # in milliseconds
                        n_intervals=0
                    ),
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
                        # html.Div(children=set_button, id='example'),
                        set_button,
                        # stop_button,
                    ], id='button_area')

                ], id='alarm_input_area_container'),
            html.Div(children=[
                html.Div(children=None, id='msg1'),
                html.Div(children=None, id='msg2'),
            ], id='msg_area'),
        ], id='alarm_area_container')

show_image =\
    html.Div(
        children=[
            html.Img(src=app.get_asset_url('Lenna.png'),
                     width="auto", height="100%")
        ], id='image_container'
    )

# show_video =\
#     html.Div(
#         children=[
#             html.Img(src="/video_feed", alt="video",
#                      width="auto", height="100%")
#         ], id='video_container'
#     )

show_video =\
    html.Div(
        children=None, id='video_container'
    )

show_video2 =\
    html.Div(
        children=None, id='video_container2'
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
            align='center', style={"height": "10vh"}
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
            align='center', style={"height": "80vh"}
        ),
        dbc.Row(
            [
                # dbc.Col(
                #     show_image,
                #     width=6,
                #     className='bg-primary',
                #     id='image_area'
                # ),
                # dbc.Col(
                #     show_video,
                #     width=6,
                #     className='bg-danger',
                #     id='video_area'
                # ),
            ],
            align='center', style={"height": "10vh"}
        ),
    ],
    fluid=True
)

stretch_layer_buttons =\
    [dbc.Button("back", outline=True,
                color="danger", className="me-1", id='back_button', href='/'),
     dbc.Button("start", outline=True,
                color="primary", className="me-1", id='video_start_button'),
     dbc.Button("stop", outline=True,
                color="danger", className="me-1", id='video_stop_button'), test_button, html.Div(id='hidden-div', style={'display': 'none'}),
     html.Div(children='a', id='msg_area2')]

s = dbc.Container(
    [
        dcc.Interval(
            id='interval_component2',
            interval=1*1000,  # in milliseconds
            n_intervals=0
        ),
        dbc.Row(
            [
                dbc.Col(
                    # show_text,
                    # back_button,
                    children=stretch_layer_buttons,
                    width=12,
                    className='bg-light',
                    id='title_area2'
                ),
            ],
            align='center', style={"height": "10vh"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    show_image,
                    width=6,
                    className='bg-primary',
                    id='image_area2'
                ),
                dbc.Col(
                    show_video2,
                    width=6,
                    className='bg-danger',
                    id='video_area2'
                ),
            ],
            align='center', style={"height": "90vh"}
        ),
    ],
    fluid=True
)
# app.layout = l2
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # html.Div(children=None, id='hidden_div_for_redirect_callback'),
    # dcc.Store(id='session', storage_type='session',
    #   data={'stretch_flag': False}),
    html.Div(id='page-content')
])


# @app.callback(Output('session', 'data'),
#               Input('test_button', 'n_clicks'),
#               State('session', 'data'), prevent_initial_call=True)
# def test(n, data):
#     print(type(data))
#     print(data)
#     data['stretch_flag'] = not data['stretch_flag']
#     return data


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return l2
    elif pathname == '/stretch':
        return s
    else:
        return '404'


# @app.callback(Output('page-content', 'children'),
#               [Input('url', 'pathname')],
#               State('session', 'data'))
# def display_page(pathname, data):
#     if pathname == '/':
#         return l2
#     elif pathname == '/stretch':
#         if data['stretch_flag'] == False:
#             return l2
#         else:
#             return s
#     # elif pathname == '/stretch':
#     #     return s
#     else:
#         return '404'


@app.callback(Output('button_area', 'children'),
              Output('msg2', 'children'),
              Input('set_button', 'n_clicks'),
              State('my-date-picker-single', 'date'),
              State('hour-picker', 'value'),
              State('minute-picker', 'value'), prevent_initial_call=True)
def set_button_pushed(set_n, date_value, hour_value, minute_value):
    if (set_n is None) or (set_n == 0):
        raise PreventUpdate
    else:
        time_str = date_value + "-"+str(hour_value)+"-"+str(minute_value)+"-0"
        set_time = dt.strptime(time_str, '%Y-%m-%d-%H-%M-%S')
        current_time = dt.now()
        if set_time > current_time:
            # alarm_dates.append(setted_time)
            global alarm_date
            print(alarm_date)
            alarm_date = set_time
            print(alarm_date)
            return cancel_button, 'You set alarm clock'
        else:
            return dash.no_update, 'Please set again'


# redirect_to_stretch = dcc.Location(
#     pathname="/stretch", id="redirect_to_stretch")


@ app.callback(Output('button_area', 'children'),
               Output('msg2', 'children'),
               Input('stop_button', 'n_clicks'),
               prevent_initial_call=True)
def cancel_button_pushed(stop_button):
    # del alarm_dates[0]
    if (stop_button is None) or (stop_button == 0):
        raise PreventUpdate
    else:
        # del alarm_dates[0]
        global alarm_date
        print(alarm_date)
        alarm_date = None
        print(alarm_date)
        return set_button, None


@ app.callback(Output('hidden-div', 'children'), Input('test_button', 'n_clicks'),
               prevent_initial_call=True)
def cancel_button_pushed(test_button):
    # del alarm_dates[0]
    if (test_button is None) or (test_button == 0):
        raise PreventUpdate
    else:
        global camera_object
        camera_object.esti_flag = not camera_object.esti_flag
        print(camera_object.all_frames)
        print(camera_object.good_counts)
        print(camera_object.get_judge())


# @app.callback(Output('session', 'data'),
#               Output('hidden_div_for_redirect_callback', 'children'),
#               Input('stop_button', 'n_clicks'),
#               State('session', 'data'),
#               prevent_initial_call=True)
# def stop_button_pushed(stop_n, data):
#     if (stop_n is None) or (stop_n == 0):
#         raise PreventUpdate
#     else:
#         data['stretch_flag'] = True
#         return data, redirect_to_stretch


@ app.callback(Output('button_area', 'children'),
               Output('msg2', 'children'),
               Input('cancel_button', 'n_clicks'),
               prevent_initial_call=True)
def cancel_button_pushed(cancel_n):
    # del alarm_dates[0]
    if (cancel_n is None) or (cancel_n == 0):
        raise PreventUpdate
    else:
        # del alarm_dates[0]
        global alarm_date
        print(alarm_date)
        alarm_date = None
        print(alarm_date)
        return set_button, 'You canceled alarm clock'


@ app.callback(Output('button_area', 'children'),
               Output('msg1', 'children'),
               Input('interval_component', 'n_intervals'))
def check_alarm(interval_n):
    # print(interval_n)
    today = dt.now()
    # if not alarm_dates:
    global alarm_date
    if alarm_date is None:
        # return '今は{0}年{1}月{2}日{3}時{4}分{5}秒です'.format(today.year, today.month, today.day, today.hour, today.minute, today.second)
        return dash.no_update, '{0}/{1}/{2} {3}:{4}:{5}'.format(today.year, str(today.month).zfill(2), str(today.day).zfill(2), str(today.hour).zfill(2), str(today.minute).zfill(2), str(today.second).zfill(2))
    else:
        # alarm_time = alarm_dates[0]
        # print(alarm_dates)
        if alarm_date > today:
            date_diff = alarm_date - today
            hours, tminute = divmod(date_diff.seconds, 3600)
            minutes, seconds = divmod(tminute, 60)
            return dash.no_update, '{0}days {1}:{2}:{3}'.format(date_diff.days, str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2))
        else:
            return stop_button, 'Alarming'


stretch_start_time = None
once_flag1 = True
once_flag2 = True


@ app.callback(Output('msg_area2', 'children'),
               Input('interval_component2', 'n_intervals'))
def a(interval_n):
    today = dt.now()
    global stretch_start_time
    if not stretch_start_time:
        stretch_start_time = today
    elapsed_time = (dt.now() - stretch_start_time).seconds
    if elapsed_time < 11:
        return 'Stretch start in ' + str(10 - elapsed_time) + ' seconds'
    elif 11 <= elapsed_time < 21:
        global once_flag1
        if once_flag1:
            camera_object.esti_flag = True
            once_flag1 = False
        return 'Stretch now!'
    else:
        global once_flag2
        if once_flag2:
            camera_object.esti_flag = False
            once_flag2 = False
        return 'result:' + str(camera_object.get_judge())

    # if(interval_n <= 10):
    #     if interval_n == 10:
    #         camera_object.esti_flag = True
    #     return 'Stretch start in ' + str(10 - interval_n) + ' seconds'
    # elif 10 < interval_n < 21:
    #     return 'Stretch now!'
    # else:
    #     if interval_n == 21:
    #         camera_object.esti_flag = False
    #     return 'result:' + str(camera_object.get_judge())


@ app.callback(Output('video_container2', 'children'),
               Input('video_start_button', 'n_clicks'), Input('video_stop_button', 'n_clicks'), prevent_initial_call=True)
def update_buttons(video_start_n, video_stop_n):
    ctx = dash.callback_context
    id = ctx.triggered[0]['prop_id'].split('.')[0]
    global camera_object
    if id == 'video_start_button':
        if camera_object is None:
            camera_object = VideoCamera()
        camera_object.start_camera()
        return html.Img(src="/video_feed", alt="video", width="auto", height="100%")
    else:
        camera_object.stop_camera()
        return None


if __name__ == '__main__':
    app.run_server(debug=True)
