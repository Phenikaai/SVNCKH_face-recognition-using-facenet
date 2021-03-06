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
import glob

# Facenet sẽ embed mặ thành vector 512 chiều, giá trị trong đấy hầu như có dạng 0,00x
# =>Khi tính khoảng cách sẽ bình phương lên nên phải nhân với 10^6
power = pow(10, 6)
frame_size = (640,480)
device =  'cpu'
namelist=[]
IMG_PATH = './data/test_images/'
DATA_PATH = './data'

# Tạo các thư mục trước
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(IMG_PATH):
    os.mkdir(IMG_PATH)
if not os.path.exists(DATA_PATH+'/embeded_data'):
        os.mkdir(DATA_PATH+'/embeded_data')
# Chuẩn hóa ảnh đưa về Tensor nằm trong khoảng [0,1]
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)


# Model Facenet
model = InceptionResnetV1(
	classify=False,
	pretrained="casia-webface"
).to(device)
model.eval()


# Lấy embed vector và tên của mỗi người
def load_faceslist():
    embeds=[]
    names = []
    for name in os.listdir(DATA_PATH+'/embeded_data'):
        names.append(name)
        embed=torch.load(DATA_PATH+'/embeded_data./'+name)
        embeds.append(embed)
    embeds=torch.cat(embeds)
    return embeds, names



# Tính toán khoảng cách của mặt với các embed đã lưu, nếu khoảng cách lớn hơn 2 => Không nhận diện được
def inference(model, face, local_embeds, threshold = 2):
    embeds = []
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds)
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)
    min_dist, embed_idx = torch.min(norm_score, dim = 1)
    # print(min_dist*power, names[embed_idx])
    if min_dist*power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

# Tách mặt từ các tọa độ có sẵn trong hình với và thêm khoảng để nhận diện tốt hơn
# Vì MTCNN lấy tọa độ ảnh mặt nên trong trường hợp mặt nghiêng, thiếu 1 phần mặt vẫn trả về tọa độ box 160*160 =>Xảy ra tọa độ âm
# Lúc này thì ta phải tăng cả margin như lúc lấy mặt và giới hạn mặt trong frame
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

# Thêm tên vào danh sách nếu khoảng cách tới 1 embed nhỏ
def addToList(score,name):
    if score < 0.2:
        with open('./attendance.txt', 'a') as f:
            # Kiểm tra tránh lặp tên liên tục
            if name not in namelist:
                f.write(f"{name}\n")
                namelist.append(name)
                f.close()


# Bắt đầu lấy dữ liệu
embeddings, names = load_faceslist()

page = st.selectbox(
    'Project',
    ('Get Data','Face Recognition'))
st.write('You selected:', page)

if page=='Get Data':
    usr_name = st.text_input("Input your name: ")
    USR_PATH = os.path.join(IMG_PATH, usr_name)
    # margin lúc lấy mặt, keep_all=False đồng nghĩa chỉ lấy 1 khuôn mặt, post_process=False để không đưa về Tensor
    mtcnn = MTCNN(margin = 20, keep_all=False, post_process=False, device = device)
    class GetFace:
        def recv(self, frame):
            frm = frame.to_ndarray(format="bgr24")
            path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")))
            # nhận diện mặt và lưu mặt luôn
            face = mtcnn(frm,save_path=path)
            cv2.putText(frm,usr_name,(50,50),cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
            return av.VideoFrame.from_ndarray(frm, format='bgr24')
    webrtc_streamer(key="key", video_processor_factory=GetFace)      
    if st.button('Embed Face'):
        for usr in os.listdir(IMG_PATH):
            if usr not in os.listdir(DATA_PATH+'/embeded_data'):
                #print(usr)
                embeds = []
                # Duyệt folder chưa ảnh mặt
                for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    # Thêm vector của mỗi mặt
                    with torch.no_grad():
                        embeds.append(model(trans(img).to(device).unsqueeze(0)))
                if len(embeds) == 0:
                    continue
                # Tính trung bình
                embedding = torch.cat(embeds).mean(axis=0, keepdim=True)
                # Lưu embed
                torch.save(embedding, DATA_PATH+'/embeded_data./'+usr)
                # Xóa ảnh mặt cũ, ta có thể dữ lại dùng cho thuật toán SVM hay Triplet Loss
                for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
                    os.remove(file)
                os.rmdir(os.path.join(IMG_PATH, usr))
                
        st.write("Success!!")
    else:
        st.write("Nothing yet")
    pp_numbers=os.listdir(DATA_PATH+'/embeded_data')
    st.write('There are %d people(s) in data'%len(pp_numbers))

elif page=='Face Recognition':
    # keep_all=True để nhận diện tất cả mặt
    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
    class FaceRecog:
        def recv(self, frame):
            frm = frame.to_ndarray(format="bgr24")
            # MTCNN sẽ trả về tọa độ các box chứa ảnh mặt
            boxes, _ = mtcnn.detect(frm)
            if boxes is not None:
                # Xét từng box mặt
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

    webrtc_streamer(key="key", video_processor_factory=FaceRecog)