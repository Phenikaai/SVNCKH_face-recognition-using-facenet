import glob
from streamlit_webrtc import webrtc_streamer
from torchvision import transforms
from PIL import Image
import streamlit as st
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from datetime import datetime
import os
import av

device =  'cpu'

IMG_PATH = './data/test_images/'
DATA_PATH = './data'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
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

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)
    
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

model.eval()
if not os.path.exists(DATA_PATH+'/embeded_data'):
    os.mkdir(DATA_PATH+'/embeded_data')

for usr in os.listdir(IMG_PATH):
    if usr not in os.listdir(DATA_PATH+'/embeded_data'):
        print(usr)
        embeds = []
        for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
            try:
                img = Image.open(file)
            except:
                continue
            with torch.no_grad():
                embeds.append(model(trans(img).to(device).unsqueeze(0)))
        if len(embeds) == 0:
            continue
        embedding = torch.cat(embeds).mean(axis=0, keepdim=True)
        torch.save(embedding, DATA_PATH+'/embeded_data./'+usr)
    
pp_numbers=os.listdir(DATA_PATH+'/embeded_data')
st.write('There are %d people(s) in data'%len(pp_numbers))