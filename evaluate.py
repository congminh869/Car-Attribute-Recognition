import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model
import numpy as np
def xnor(lst1, lst2):
    arr_result = []
    for count, i in enumerate(lst1):
        if i == lst2[count]:
            arr_result.append(1)
        else:
            arr_result.append(0)
    return arr_result
with open('./doc/my_label.json', 'r', encoding = 'utf-8') as labels:
    label = json.load(labels)['attribute']
    label= list(label.values())
label.sort()
with open('./doc/final_Duke.json', 'r', encoding = 'utf-8') as duke:
    duke_att = json.load(duke)
    Duke = dict()
    for i in duke_att:
        num_arr = []
        for j in label:
            if j not in duke_att[i].keys():
                num_arr.append(0)
            else:
                num_arr.append(duke_att[i][j])
        Duke[i] = num_arr
with open('./doc/final_Market.json', 'r', encoding = 'utf-8') as market:
    market_att = json.load(market)
    Market = dict()
    for i in market_att:
        num_arr = []
        for j in label:
            if j not in market_att[i].keys():
                num_arr.append(0)
            else:
                num_arr.append(market_att[i][j])
        Market[i] = num_arr
######################################################################
# Settings
# ---------
num_label = 44
num_id = 250
transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
use_id = False
######################################################################
# Argument
# ---------
model_name = '{}_nfc_id'.format('resnet50') if use_id else '{}_nfc'.format('resnet50')
# num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]
# print(num_label, num_id)
num_label, num_id = (43, 57184)
# Model and Data
# ---------
def load_network(network):
    save_path = os.path.join('./checkpoints', 'person.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network
def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src
print(num_label, num_id)
model = get_model(model_name, num_label, use_id=False, num_id=num_id)
model = load_network(model)
model.eval()
num = 0
result = np.empty((0,len(label)), int)
for name in ['Duke', 'Market']:
    for root, dirs, files in os.walk('D:/WORKIT/Person-Attribute-Recognition-MarketDuke/dataset'):
        for file in files:
            if file.endswith('.jpg'):
                if root.find(name) != -1:
                    num += 1
                    if num % 100 == 0:
                        print(num)
                    if num == 50000:
                        break
                    image_path = os.path.join(root, file)
                    ids = file[:4]
                    src = load_image(image_path)
                    out = model.forward(src)
                    pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
                    pred = pred.tolist()[0]
                    # any(pred[inter] = 1 if i else pred[inter] = 0 for inter, i in enumerate(pred))
                    for inter, i in enumerate(pred):
                        if i:
                            pred[inter] = 1
                        else:
                            pred[inter] = 0
                    compare = xnor(globals()[name][ids],  pred)
                    compare = np.array(compare)
                    result = np.vstack ((result, compare) )
        if num == 50000:
            break
    if num == 50000:
        break
np.save("result.npy", result)
row, colum = result.shape
for count, i in enumerate(label):
    check = result[:, count]
    number = 0
    for j in check:
        if j == 1:
            number += 1
    print(i, round(number/len(check), 4))