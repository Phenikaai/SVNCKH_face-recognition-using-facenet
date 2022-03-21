import glob
from sympy import im
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np

IMG_PATH = './project/data/test_images'
DATA_PATH = './project/data'

device =  'cpu'
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
print('Update complete!, There are %d people(s) in data'%len(pp_numbers))