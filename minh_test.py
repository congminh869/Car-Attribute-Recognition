import os
import glob
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model


######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':18, 'duke':23,} #20
num_ids_dict = { 'market':34502, 'duke':702 }#301

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='test_sample/img6.jpg', type=str,help='Path to test image')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
# print('model_name = ',model_name)
num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]
# print('num_label = ', num_label)
# print('num_id = ',num_id)


######################################################################
# Model and Data
# ---------
def load_network(network):
    # save_path = os.path.join('./checkpoints', args.dataset, model_name, 'net_last.pth')
    save_path = '/home/minh/Documents/minh/mqsolutions/detec_color/Person-Attribute-Recognition-MarketDuke/checkpoints/market/resnet50_nfc/net_last.pth'
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    srcs = []
    names = []
    for name in glob.glob(path+'*.jpg'):
        # print(name)
        src = Image.open(name)
        src = transforms(src)
        src = src.unsqueeze(dim=0)
        srcs.append(src)
        names.append(name.split('/')[-1])
    return srcs, names


model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model = load_network(model)
model.eval()

# name_img_test = names[0]
# print('name: ',name_img_test)
# print(names)

######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        # with open('./doc/label.json', 'r') as f:
            # self.label_list = json.load(f)[dataset]


        # self.label_list = ['red','orange','yellow','green','blue','purple','brown','pink','gray','white','black','Mini','Midsize','Large','bus','truck','car','Coach','Taxi','Lorry']
        self.label_list = ['black', 'blue', 'brown', 'bus', 'car', 'coach', 'gray', 'green', 'large', 'lorry', 'medium', 'orange', 'pink', 'red', 'sandy', 'truck', 'white', 'yellow']

        self.attribute_dict = {
        "black": ['black',['no','1']],
        "blue": ['blue',['no','1']],
        "brown": ['brown',['no','1']],
        "bus": ['bus',['no','1']],
        "car": ['car',['no','1']],
        "coach": ['coach',['no','1']],
        "gray": ['gray',['no','1']],
        "green": ['green',['no','1']],
        "large": ['large',['no','1']],
        "lorry": ['lorry',['no','1']],
        "medium": ['medium',['no','1']],
        "orange": ['orange',['no','1']],
        "pink": ['pink',['no','1']],
        "red": ['red',['no','1']],
        "sandy": ['sandy',['no','1']],
        "truck": ['truck',['no','1']],
        "white": ['white',['no','1']],
        "yellow": ['yellow',['no','1']]
        }
        self.dataset = dataset
        # print('dataset = ', dataset)
        self.num_label = len(self.label_list)
        # print('num_label = ', num_label)

    def decode(self, pred):
        result = []
        pred = pred.squeeze(dim=0)
        count  = 0
        att = []
        color = [0,1,2,6,7,11,12,13,16,17]
        size  = [8,10]
        types = [3,4,5,9,14,15]
        name_size = ''
        name_type = ''
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            
            if chooce[pred[idx]]=='1':
                count +=1
                # print(idx)
                result.append(name)
                # print(name)
                # print('{}: {}'.format(name, chooce[pred[idx]]))
        if count == 4 :
            name_color = []
            for idx in range(self.num_label):
                name, chooce = self.attribute_dict[self.label_list[idx]]                
                if chooce[pred[idx]]=='1':
                    if idx in color:
                        name_color.append(name)
                    elif idx in size:
                        name_size = name
                    else:
                        name_type = name
            name_img_final = name_type+'_'+name_color[0]+'-'+name_color[1]+'_'+name_size
            # print(name_img_final)
        else:
            for idx in range(self.num_label):
                name, chooce = self.attribute_dict[self.label_list[idx]]                
                if chooce[pred[idx]]=='1':
                    if idx in color:
                        name_color = name
                    elif idx in size:
                        name_size = name
                    else:
                        name_type = name
            name_img_final = name_type+'_'+name_color+'_'+name_size
            # print(name_img_final)

                    # count +=1
                    # result.append(name)
        # print('count = ',count)
        return result, name_img_final



path_img = '/home/minh/Documents/minh/mqsolutions/detec_color/Person-Attribute-Recognition-MarketDuke/test_sample/img_detect/'
srcs, names = load_image(path_img)
# src = srcs[0]

a = 0
for i in range(len(srcs)):
    if not args.use_id:
        out = model.forward(srcs[i])
    else:
        out, _ = model.forward(srcs[i])

    # print('out = ',out)
    try:
        pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
        # print(names[i])
        Dec = predict_decoder(args.dataset)
        result, name_img_final = Dec.decode(pred)
        name_img_final = name_img_final+'_'+names[i].split('_')[0]+'.jpg'
        
        # print(name_img_final)
        # if a == 3:
        #     break
        a+=1
        print(a)
        os.rename(path_img + names[i],path_img + name_img_final) 
        print('------------')
    except:
        print(names[i])
#coach_gray_medium_1