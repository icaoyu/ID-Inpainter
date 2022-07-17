import argparse
import numpy as np
from PIL import Image
import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from lib.core.config import config
from lib.core.config import update_config
from lib.core.lfw_eval import eval as lfw_eval
from lib.core.lfw_eval import eval_roc
from lib.core.lfw_eval import extractDeepFeatureSingle, cosine_similarity
from lib.datasets.dataset import LFW_Image
# from model.CBAM import CBAMResNet
from arcface.face_model import MobileFaceNet
from utils import Visualizer


# setup random seed
torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch End2End Occluded Face')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--binary_thres', help='thres for binary mask', type=float)
    parser.add_argument('--soft_binary', help='whether use soft binary mask', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--pretrained', help='whether use pretrained model', type=str)
    parser.add_argument('--debug', help='whether debug', default=0, type=int)
    parser.add_argument('--model', help=' model name', type=str)
    parser.add_argument('--factor', help='factor of mask',  type=float)
    parser.add_argument('--ratio', help='ratio of masked img for training', default=4, type=int)
    args = parser.parse_args()

    return args

def plot_roc(occ_paths, visualizer, append=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------------model----------------------------------------
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    test_loaders = []

    # occ_paths = ['data_occ/lfws1/random_block_occ/', 'data_occ/lfws1/random_block_ca_fill/',
    #              'data_occ/lfws1/random_block_pic_fill/', 'data_occ/lfws1/random_block_ca_id_fill/']
    # occ_paths = ['data_occ/lfws1/random_part_occ/', 'data_occ/lfws1/random_part_ca_fill/',
    #              'data_occ/lfws1/random_part_pic_fill/', 'data_occ/lfws1/random_part_ca_id_fill/']
    for p in occ_paths:

        config.DATASET.LFW_PATH = 'data_occ/lfws/lfw_align_112/'
        config.DATASET.LFW_OCC_PATH = p
        config.DATASET.LFW_PAIRS = 'data_occ/lfws/pairs.txt'

        test_loader = torch.utils.data.DataLoader(
            LFW_Image(config, test_transform),
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=config.TEST.SHUFFLE,
            num_workers=config.TEST.WORKERS,
            pin_memory=True)

        test_loaders.append(test_loader)

    model_list = ['Arc. ']

    for model_name in model_list:
        recognizer = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
        recognizer.load_state_dict(torch.load('arcface/MobileFace_Net', map_location=lambda storage, loc: storage))

        recognizer.to(device)
        recognizer.eval()
        # visualizer = Visualizer()
        eval_roc(recognizer, model_name, config, test_loaders, 'temp', 0, visualizer=visualizer, append=append)

def plot():
    occ_paths = ['data/LFWs/20220712/random_block_occ/', 'data/LFWs/20220712/random_block_ca_fill/',
                              'data/LFWs/20220712/random_block_pic_fill/', 'data/LFWs/20220712/random_block_ca_id_fill/']
    visualizer = Visualizer()
    plot_roc(occ_paths, visualizer, True)

    # from_roc(paths, visualizer, True)





def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------------model----------------------------------------
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # config.DATASET.LFW_PATH = 'data_occ/CFP-FP/cfp-crop/Data/Images/'
    # config.DATASET.LFW_OCC_PATH = 'data_occ/CFP-FP/ca_id_fill_mask/Data/Images/'
    # config.DATASET.LFW_PAIRS = 'data_occ/CFP-FP/cfp-crop/Protocol/Pair_list_F.txt'
    # config.DATASET.LFW_CLASS = 6000
    #
    # test_loader = torch.utils.data.DataLoader(
    #     CFP_Image(config, test_transform),
    #     batch_size=config.TEST.BATCH_SIZE,
    #     shuffle=config.TEST.SHUFFLE,
    #     num_workers=config.TEST.WORKERS,
    #     pin_memory=True)


    config.DATASET.LFW_PATH = 'data/LFWs/lfw_112/'
    config.DATASET.LFW_OCC_PATH = 'data/LFWs/20220712/random_part_ca_id_fill/'
    config.DATASET.LFW_PAIRS = 'data/LFWs/pairs.txt'


    test_loader = torch.utils.data.DataLoader(
        LFW_Image(config, test_transform),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.TEST.WORKERS,
        pin_memory=True)

    model_root = 'pretrained/'
    # model_list = ['model_p5_w1_9938_9470_6503.pth.tar',
    #               'model_p4_baseline_9938_8205_3610.pth.tar']
    model_list = ['arcface']
    # model_list = [
        # 'model_best_p5_w0.pth.tar',
        # 'model_best_p5_w1.pth.tar',
        # 'model_best_p5_occ.pth.tar'
    # ]
    for model_name in model_list:
        # pattern = int(model_name[model_name.find('p')+1])
        # num_mask = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, pattern))
        # model = LResNet50E_IR_FPN(num_mask=num_mask)
        # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

        # recognizer = CBAMResNet(50, feature_dim=512, mode='ir')  # resnet18
        # recognizer.load_state_dict(torch.load('./arcface/model_ir_se50.pth', map_location=device))
        recognizer = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
        recognizer.load_state_dict(torch.load('arcface/MobileFace_Net', map_location=lambda storage, loc: storage))
        # print('MobileFaceNet face detection model generated')

        # detect_model.eval()

        recognizer.to(device)
        recognizer.eval()
        lfw_eval(recognizer, model_name, config, test_loader, 'temp', 0)


def get_distance(img1_path, img2_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recognizer = CBAMResNet(50, feature_dim=512, mode='ir')  # resnet18
    # recognizer = DataParallel(recognizer)
    recognizer.load_state_dict(torch.load('./arcface/Iter_64000_net.pth', map_location=device))

    recognizer.to(device)
    recognizer.eval()

    with open(img1_path, 'rb') as f:
        img1 = Image.open(f).convert('RGB')

    with open(img2_path, 'rb') as f:
        img2 = Image.open(f).convert('RGB')

    # print(img1.size)
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    img1 = transform(img1).to('cuda').unsqueeze(0)
    img2 = transform(img2).to('cuda').unsqueeze(0)
    # print(img1.size())
    f1 = extractDeepFeatureSingle(img1, recognizer)
    f2 = extractDeepFeatureSingle(img2, recognizer)

    # print(f2.size())
    # distance = cosine_similarity(f1.detach(), f2.detach())
    f1 = f1.detach().numpy()
    f2 = f2.detach().numpy()
    # print(f1.shape, f2.shape)

    A = np.sum(f1 * f2)
    B = np.linalg.norm(f1, axis=0) * np.linalg.norm(f2, axis=0) + 1e-5
    distance = A / B
    return 1-distance

if __name__ == '__main__':
    # plot()
    main()



