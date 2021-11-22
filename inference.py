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
parser.add_argument('image_path', help='Path to test image')
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
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src


model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model = load_network(model)
model.eval()

src = load_image(args.image_path)

######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        # with open('./doc/label.json', 'r') as f:
            # self.label_list = json.load(f)[dataset]


        # self.label_list = ['red','orange','yellow','green','blue','purple','brown','pink','gray','white','black','Mini','Midsize','Large','bus','truck','car','Coach','Taxi','Lorry']
        self.label_list = ['black', 'blue', 'brown', 'bus', 'car', 'coach', 'gray', 'green', 'large', 'lorry', 'medium', 'orange', 'pink', 'red', 'sandy', 'truck', 'white', 'yellow']


        # print('self.label_list = ', self.label_list)
        # with open('./doc/attribute.json', 'r') as f:
        #     self.attribute_dict = json.load(f)[dataset]
        #     print('self.attribute_dict = ', self.attribute_dict)
        # self.attribute_dict = {
        # 'red':['red',['no','1']],
        # 'orange':['orange',['no','1']],
        # 'yellow':['yellow',['no','1']],
        # 'green':['green',['no','1']],
        # 'blue':['blue',['no','1']],
        # 'purple':['purple',['no','1']],
        # 'brown':['brown',['no','1']],
        # 'pink':['pink',['no','1']],
        # 'gray':['gray',['no','1']],
        # 'white':['white',['no','1']],
        # 'black':['black',['no','1']],
        # 'Mini':['Mini',['no','1']],
        # 'Midsize':['Midsize',['no','1']],
        # 'Large':['Large',['no','1']],
        # 'bus':['bus',['no','1']],
        # 'truck':['truck',['no','1']],
        # 'car':['car',['no','1']],
        # 'Coach':['Coach',['no','1']],
        # 'Taxi':['Taxi',['no','1']],
        # 'Lorry':['Lorry',['no','1']]
        # } 

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
        '''
            {'bag': ['carrying bag', ['no', 'yes']], 
            'upred': ['color of upper-body clothing', [None, 'red']], 
            'upblue': ['color of upper-body clothing', [None, 'blue']], 
            'hat': ['wearing hat', ['no', 'yes']], 
            'downgreen': ['color of lower-body clothing', [None, 'green']], 
            'downbrown': ['color of lower-body clothing', [None, 'brown']], 
            'upyellow': ['color of upper-body clothing', [None, 'yellow']], 
            'up': ['sleeve length', ['long sleeve', 'short sleeve']], 
            'upgreen': ['color of upper-body clothing', [None, 'green']], 
            'handbag': ['carrying handbag', ['no', 'yes']], 
            'downgray': ['color of lower-body clothing', [None, 'gray']], 
            'clothes': ['type of lower-body clothing', ['dress', 'pants']], 
            'adult': ['age', [None, 'adult']], 
            'downblack': ['color of lower-body clothing', [None, 'black']], 
            'backpack': ['carrying backpack', ['no', 'yes']], 
            'downwhite': ['color of lower-body clothing', [None, 'white']], 
            'upblack': ['color of upper-body clothing', [None, 'black']], 
            'gender': ['gender', ['male', 'female']], 
            'downyellow': ['color of lower-body clothing', [None, 'yellow']], 
            'downpink': ['color of lower-body clothing', [None, 'pink']], 
            'old': ['age', [None, 'old']], 
            'down': ['length of lower-body clothing', ['long lower body clothing', 'short']], 
            'uppurple': ['color of upper-body clothing', [None, 'purple']], 
            'downpurple': ['color of lower-body clothing', [None, 'purple']], 
            'young': ['age', [None, 'young']], 
            'teenager': ['age', [None, 'teenager']], 
            'hair': ['hair length', ['short hair', 'long hair']], 
            'downblue': ['color of lower-body clothing', [None, 'blue']], 
            'upgray': ['color of upper-body clothing', [None, 'gray']], 
            'upwhite': ['color of upper-body clothing', [None, 'white']]}

        '''
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

