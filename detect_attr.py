import os
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
parser.add_argument('--image_path', default='/home/minh/Documents/mqsolutions/Person-Attribute-Recognition-MarketDuke/test_sample/img4.jpgt', help='Path to test image')
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
    save_path = '/home/minh/Documents/minh/mqsolutions/detec_color/Person-Attribute-Recognition-MarketDuke/checkpoints/market/resnet50_nfc/net_30.pth'
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src


model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model = load_network(model)
model.eval()

image_path1 = '/home/minh/Documents/minh/mqsolutions/detec_color/Person-Attribute-Recognition-MarketDuke/test_sample/img4.jpg'
src = load_image(image_path1)
######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        # with open('./doc/label.json', 'r') as f:
            # self.label_list = json.load(f)[dataset]


        # self.label_list = ['red','orange','yellow','green','blue','purple','brown','pink','gray','white','black','Mini','Midsize','Large','bus','truck','car','Coach','Taxi','Lorry']
        self.label_list = ['red',
                         'orange',
                         'yellow',
                         'green',
                         'blue',
                         'purple',
                         'brown',
                         'pink',
                         'gray',
                         'white',
                         'black',
                         'Mini',
                         'Medium',
                         'Large',
                         'bus',
                         'truck',
                         'car',
                         'coach',
                         'taxi',
                         'lorry']


        self.attribute_dict = {
            "red": ["red",["no","1"]],
            "orange": ["orange",["no","1"]],
            "yellow": ["yellow",["no","1"]],
            "green": ["green",["no","1"]],
            "blue": ["blue",["no","1"]],
            "purple": ["purple",["no","1"]],
            "brown": ["brown",["no","1"]],
            "pink": ["pink",["no","1"]],
            "gray": ["gray",["no","1"]],
            "white": ["white",["no","1"]],
            "black": ["black",["no","1"]],
            "Mini": ["Mini",["no","1"]],
            "Medium": ["Medium",["no","1"]],
            "Large": ["Large",["no","1"]],
            "bus": ["bus",["no","1"]],
            "truck": ["truck",["no","1"]],
            "car": ["car",["no","1"]],
            "coach": ["coach",["no","1"]],
            "taxi": ["taxi",["no","1"]],
            "lorry": ["lorry",["no","1"]]
        }
        self.dataset = dataset
        # print('dataset = ', dataset)
        self.num_label = len(self.label_list)
        # print('num_label = ', num_label)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))


if not args.use_id:
    out = model.forward(src)
else:
    out, _ = model.forward(src)

pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5

Dec = predict_decoder(args.dataset)
Dec.decode(pred)