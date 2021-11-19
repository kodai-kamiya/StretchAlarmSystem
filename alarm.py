#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:24:34 2021

@author: kamiyakoudai
"""

import pygame.mixer
import time
import schedule

# from mutagen.mp3 import MP3 as mp3
# from pygame.locals import *
# import os
# import datetime
# import cv2
# import sys


def Alarm():
    print("時間です")
    job()
    exit()

def job():
    while True:
        filename = 'GoodMorning.mp3' #再生したいmp3ファイル
        pygame.mixer.init()
        pygame.mixer.music.load(filename) #音源を読み込み
        pygame.mixer.music.play(-1) #再生開始。1の部分を変えるとn回再生(その場合は次の行の秒数も×nすること)
        input()
        pygame.mixer.music.stop() #終了


        
#目覚まし設定時間取得
print("目覚ましをセットする時間を指定してください")
hour = input("時間（hour）：")
minute = input("分（minute）：")
target = f"{hour.zfill(2)}:{minute.zfill(2)}"
print(target+"にアラームをセットしました")

#アラーム時間設定
schedule.every().day.at(target).do(job)
  
while True:
  schedule.run_pending()
  time.sleep(1)
