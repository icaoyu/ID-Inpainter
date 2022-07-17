
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
import argparse
import json
import os
import time
import shutil
import logging
import numpy as np
import pylab as p
from PIL import Image

import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import lib.core.utils as utils
from lib.core.config import config
from lib.core.config import update_config

from lib.core.lfw_eval import extractDeepFeatureSingle, extractDeepFeature20
from lib.datasets.dataset import LFW_Image
from model.CBAM import CBAMResNet
from arcface.face_model import MobileFaceNet, l2_norm

import matplotlib.cm as cm
from sklearn.preprocessing import Normalizer,MinMaxScaler, StandardScaler



# setup random seed
torch.manual_seed(0)
np.random.seed(0)

def extractFeature(model, imgs):
    model.eval()
    # print(imgs.size())
    fc = extractDeepFeature20(imgs, model)
    return fc

def getImgSet(path):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    img_set = []
    for img in os.listdir(path):
        with open(path+img, 'rb') as f:
            img = Image.open(f).convert('RGB')
            ## FROM
            # img = img.crop((8, 0, 104, 112))
        img = transform(img)
        # print(img.size())
        img_set.append(img)
    return torch.stack(img_set)

if __name__ == '__main__':
    root_dir = 'tsne/'
    class_list = [str(c) for c in range(1, 6)]
    colors = ['red', 'blue', 'aqua', 'green', 'purple']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
    gpus = range(len(gpus))
    # print(imgs.size())
    # arcFace
    # recognizer = CBAMResNet(50, feature_dim=512, mode='ir')  # resnet18
    # # recognizer = DataParallel(recognizer)
    # recognizer.load_state_dict(torch.load('./arcface/Iter_64000_net.pth', map_location=device))
    recognizer = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    recognizer.load_state_dict(torch.load('arcface/MobileFace_Net', map_location=lambda storage, loc: storage))

    recognizer.to(device)
    recognizer.eval()
    #
    # FROM
    # name = 'model_p5_w1_9938_9470_6503.pth.tar'
    # pattern = int(name[name.find('p') + 1])
    # num_mask = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, pattern))
    # # print(num_mask)
    # recognizer = LResNet50E_IR_FPN(num_mask=num_mask)
    # recognizer = torch.nn.DataParallel(recognizer, device_ids=gpus).cuda()
    # # recognizer = DataParallel(recognizer)
    # checkpoint = torch.load('./FROM/model_p5_w1_9938_9470_6503.pth.tar')
    # # checkpoint = torch.load('./FROM/model_p4_baseline_9938_8205_3610.pth.tar')
    #
    # state_dict = checkpoint['state_dict']
    # if isinstance(recognizer, torch.nn.DataParallel):
    #     recognizer.module.load_state_dict(state_dict, strict=False)
    # else:
    #     recognizer.load_state_dict(state_dict, strict=False)

    # # recognizer.load_state_dict(torch.load()['model_state_dict'])
    # # recognizer.to(device)
    # recognizer.eval()

    ## unocc
    features2d = []

    features = []
    for c in class_list:
        # print(c)
        imgs = getImgSet('tsne/facecrub112/'+c+'/')
        # imgs = getImgSet('tsne/up_face_occ/'+c+'/')

        fc = extractFeature(recognizer, imgs)
        # print(fc.size())
        fc = fc.detach().numpy()
        features.append(fc)
    features = np.vstack(features)
    # print(features.shape)

    feature2d = TSNE(n_components=2, init='pca').fit_transform(features)
    # feature2d = StandardScaler().fit_transform(feature2d)

    feature2d = Normalizer().fit_transform(feature2d)
    # print(feature2d.shape)

    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(5):
        # f = feature2d[i*20:(i+1)*20, :]
        # plt.scatter(f[:,0], f[:,1], color=colors[i])
        ax.scatter(feature2d[i*20:(i+1)*20, 0], feature2d[i*20:(i+1)*20, 1], color=colors[i])
    # plt.show()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    # plt.show()
    plt.savefig('tsne/fig/arc_unocc.png')
    # #






