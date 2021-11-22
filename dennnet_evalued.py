import os
import json
import glob
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model
import numpy as np
import time
def xnor(lst1, lst2):
    arr_result = []
    for count, i in enumerate(lst1):
        if i == lst2[count] == 0:
            arr_result.append(None)
        elif i == lst2[count] == 1:
            arr_result.append(1)
        elif i != lst2[count]:
            arr_result.append(0)
    return arr_result

num_label = 20

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_id = False
model_name = '{}_nfc_id'.format('densenet121') if use_id else '{}_nfc'.format('densenet121')
# num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]
# print(num_label, num_id)
num_label, num_id = (20,1)

# def load_network(network):
#     save_path = os.path.join('/home/minh/Documents/minh/mqsolutions/detec_color/Person-Attribute-Recognition-MarketDuke/checkpoints/market/resnet50_nfc', 'net_30.pth')
#     network.load_state_dict(torch.load(save_path, map_location="cuda:0"))
#     print('Resume model from {}'.format(save_path))
#     return network

# def load_network(network):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     save_path = os.path.join('/home/minh/Documents/minh/mqsolutions/detec_color/Person-Attribute-Recognition-MarketDuke/checkpoints/market/resnet50_nfc', 'net_30.pth')
#     network.load_state_dict(torch.load(save_path, map_location=device)) 
#     network.to(device)
#     print('Resume model from {}'.format(save_path))
#     return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src.cuda()

model = get_model(model_name, num_label, use_id=False, num_id=num_id)
# model = load_network(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_path = os.path.join('./checkpoints/market/densenet121_nfc', 'net_50.pth')
model.load_state_dict(torch.load(save_path, map_location=device)) 
model.to(device)
model.eval()
model.cuda()
num = 0


with open('./doc/final_label.json', 'r', encoding = 'utf-8') as labels:
    label = json.load(labels)
    label= list(label.values())

path_json = './dataset/data_set_minh10/label_train10.json'
with open(path_json) as f:
    label_train = json.load(f)

    
path = './dataset/data_set_minh10/train/'

list_dir = os.listdir(path)
print(list_dir)
result = np.empty((0,20), int)


path_name_all_imgs = []
for name_folder in list_dir:
    path_folder = path + name_folder +'/'
    for name in glob.glob(path_folder+'*.jpg'):
        path_name_all_imgs.append(name)


import random
random.shuffle(path_name_all_imgs)

start = time.time()
name_imgs = []
k = 0
if True:
    for name in path_name_all_imgs:
        ids = name.split('/')[-1].split('.')[0]
        
        image_path = name
        ids=name.split('/')[-1].split('.')[0]
        src = load_image(image_path)
        out = model.forward(src)
        pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
        pred = pred.tolist()[0]

        for inter, i in enumerate(pred):
            if i:
                pred[inter] = 1
            else:
                pred[inter] = 0
#         print(pred)
#         print(label_train[ids])
#         print('------------------')
#         print(type(pred))
        compare = xnor(label_train[ids],  pred)
        compare = np.array(compare)
        result = np.vstack((result, compare))  
        k+=1
        if k%1000==0:
            stop = time.time()
            t = stop-start
            print(f'{k} :  {t}')
            start = time.time()
            name_save = "./dataset/data_set_minh10/numpy/result_dennet_2"+str(k)+".npy"
            np.save(name_save, result)
            a = 0
            for count, i in enumerate(label):
                check = result[:, count]
                number = 0
                total = 0
                for j in check:
                    if j == 1:
                        number += 1
                        total += 1
                    elif j == 0:
                        total += 1
                try:
                    print(i, round(number/total, 4))
                    a = a + round(number/total, 4)
                except Exception as e:
                    print("An exception occurred: ", e) 
                    print(i)
            print(' acc = ', a/19)
            print('------------------------------------------------------')