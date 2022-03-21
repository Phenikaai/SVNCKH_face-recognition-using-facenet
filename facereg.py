from streamlit_webrtc import webrtc_streamer
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image
from torchvision import transforms
from datetime import datetime
import os
import torch
import streamlit as st
import av
import cv2
power = pow(10, 6)
device = 'cpu'

model = InceptionResnetV1(
	classify=False,
	pretrained="casia-webface"
).to(device)
model.eval()

mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

frame_size = (640,480)
DATA_PATH = './project/data'
namelist=[]
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)

def load_faceslist():
    embeds=[]
    names = []
    for name in os.listdir(DATA_PATH+'/embeded_data'):
        names.append(name)
        embed=torch.load(DATA_PATH+'/embeded_data./'+name)
        embeds.append(embed)
    embeds=torch.cat(embeds)
    return embeds, names

def inference(model, face, local_embeds, threshold = 3):
    embeds = []
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds)
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)
    min_dist, embed_idx = torch.min(norm_score, dim = 1)
    print(min_dist*power, names[embed_idx])
    if min_dist*power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face

def addToList(score,name):
    if score < 0.2:
        with open('project/attendance.txt', 'a') as f:
            if name not in namelist:
                f.write(name+' '+str(datetime.now())+"\n")
                namelist.append(name)
                f.close()
embeddings, names = load_faceslist()

class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        boxes, _ = mtcnn.detect(frm)
        print(boxes)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int,box.tolist()))
                face = extract_face(bbox, frm)
                idx, score = inference(model, face, embeddings)
                if idx != -1:
                    frm = cv2.rectangle(frm, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                    score = torch.Tensor.cpu(score[0]).detach().numpy()*power
                    frm = cv2.putText(frm, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                    addToList(score,names[idx])
                else:
                    frm = cv2.rectangle(frm, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                    frm = cv2.putText(frm,"unknow", (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor)