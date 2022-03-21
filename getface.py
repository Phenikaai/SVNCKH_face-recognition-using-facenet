from numpy import save
from streamlit_webrtc import webrtc_streamer
import streamlit as st
import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os
import av

device =  'cpu'

IMG_PATH = './data/test_images'
if not os.path.exists(IMG_PATH):
    os.mkdir(IMG_PATH)
count = 1
usr_name = st.text_input("Input your name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
mtcnn = MTCNN(margin = 20, keep_all=False, post_process=False, device = device)
class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")+str(count)))
        face = mtcnn(frm,save_path=path)
        cv2.putText(frm,usr_name,(50,50),cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
        return av.VideoFrame.from_ndarray(frm, format='bgr24')
webrtc_streamer(key="key", video_processor_factory=VideoProcessor)