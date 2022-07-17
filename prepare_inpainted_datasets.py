import argparse
import os
import random
# from random import random
from PIL import Image
import torchvision.transforms as transforms
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from fusor_net import Fusor_Net
import cv2
import numpy as np
from ca_inpainter.model.camodel import create_ca_model
from ca_inpainter.utils.tools import get_config

def occlist_reader(fileList):
    occList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            occPath = line.strip()
            occList.append(occPath)
    return occList

def get_occ_boxs_v2():
    # 112x112
    left_eye = (18, 30, 58, 70)
    right_eye = (58, 30, 98, 70)
    twoeyes = (18, 30, 98, 70)
    nose = (42, 48, 70, 86)
    mouth = (33, 82, 83, 107)
    down_face = (16, 60, 100, 110)
    up_face = (16, 8, 100, 72)
    left_face = (16, 8, 58, 100)
    right_face = (58, 8, 100, 100)
    center_block = (32, 32, 80, 80)
    # RMFD = (0, 60, 112, 112)

    boxes = [left_eye, right_eye, twoeyes, nose, mouth, down_face, up_face, left_face, right_face, center_block]
    names = ['left_eye', 'right_eye', 'twoeyes', 'nose', 'mouth', 'down_face', 'up_face', 'left_face', 'right_face', 'center_block']
    return boxes, names

def blending(content_img, output, mask):
    '''
    :param content_img: Image type with size(3,112,112)
    :param output: Image type with size(3,112,112)
    :param mask: (3,112,112), wiht 0 and 1
    :return: Image type with size(3,112,112)
    '''
    obj_img = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    bg_img = cv2.cvtColor(np.array(content_img), cv2.COLOR_RGB2BGR)
    mask = (1 - mask) * 255
    mask = mask.astype(np.uint8)
    monoMaskImage = cv2.split(mask)[0]
    br = cv2.boundingRect(monoMaskImage)  # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
    out_img = cv2.seamlessClone(obj_img.astype(np.uint8), bg_img.astype(np.uint8), mask.astype(np.uint8), centerOfBR,
                                cv2.NORMAL_CLONE)
    out_img = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    return out_img


def create_dir(dir):
    dirname = os.path.dirname(dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def generate_mask(img,box):
    #img:(3,112,112)
    # print(img.size())
    # print(box)
    mask = torch.ones_like(img)
    start_x, start_y, end_x, end_y = box
    mask[:, start_y:end_y, start_x:end_x] = 0
    return mask

def get_box(mask_type):
    # based on size(112,112)
    if mask_type == 'random_block':
        s = (1,112,112)
        # s = (1, 112, 96)
        holesize = (64,64)
        N_mask = random.randint(1, 5)
        limy = s[1] - s[1] / (N_mask + 1)
        limx = s[2] - s[2] / (N_mask + 1)
        # for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        if y > s[1] - holesize[0]:
            y = s[1] - holesize[0]
        if x > s[2] - holesize[1]:
            x = s[2] - holesize[1]
        range_x = x + holesize[1]
        range_y = y + holesize[0]
        box = int(x), int(y), int(range_x), int(range_y)
    else:
        boxs, names = get_occ_boxs_v2()
        if mask_type == 'random_part':
            box = random.choice(boxs)
        else:
            box = boxes[names.index(mask_type)]
    return box


def generate_occ(img,occList, Occluders, box):
    occ_path = random.choice(occList)
    occ = PIL_Reader(os.path.join(Occluders, occ_path))
    W, H = img.size
    occ_w, occ_h = box[2] - box[0], box[3] - box[1]
    new_w, new_h = min(W - 1, occ_w), min(H - 1, occ_h)
    occ = occ.resize((new_w, new_h))
    img.paste(occ, box)
    return img

def gettransform():
    transform_list = [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    return transform

def createDirs(root,mask,namelist):
    for dir in namelist:
        create_dir(os.path.join(root,mask+'_'+dir))

def PIL_Reader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def generate_pic_fill(model,img_path,box,device):
    target_img = trans(Image.open(img_path))
    mask = generate_mask(target_img, box)  # (3,112,112) ,
    mask = mask.unsqueeze(0).to(device)
    target_img = target_img.unsqueeze(0).to(device)
    masked_img = target_img*mask
    output = model.infer(masked_img, mask)
    output = (output + 1) / 2
    output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
    return output


def generate_id_fill(model,img_path,box,device):
    target_img = trans(Image.open(img_path))
    mask = generate_mask(target_img, box)  # (3,112,112) ,
    mask = mask.unsqueeze(0).to(device)
    target_img = target_img.unsqueeze(0).to(device)
    source_img = target_img * mask - (1 - mask)
    with torch.no_grad():
        output, z_id, output_z_id, feature_map, output_feature_map = model.forward(target_img, source_img)
    output = (output + 1) / 2
    output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
    target_temp = (target_img + 1) / 2
    target_temp = transforms.ToPILImage()(target_temp.cpu().squeeze().clamp(0, 1))
    mask_temp = np.array((mask.cpu().squeeze(axis=0).clamp(0, 1))).transpose((1, 2, 0))
    fill_img = blending(target_temp, output, mask_temp)
    return fill_img


def generate_inpainting_swap(imodel,smodel,img_path,box,device):
    target_img = trans(Image.open(img_path))
    mask = generate_mask(target_img, box)  # (3,112,112) ,
    mask = mask.unsqueeze(0).to(device)
    target_img = target_img.unsqueeze(0).to(device)
    source_img = target_img * mask
    inpaint_output = imodel.infer(source_img, mask)
    with torch.no_grad():
        output, z_id, output_z_id, feature_map, output_feature_map = smodel.forward(inpaint_output, source_img)
    output = (output + 1) / 2
    output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
    target_temp = (target_img + 1) / 2
    target_temp = transforms.ToPILImage()(target_temp.cpu().squeeze().clamp(0, 1))
    mask_temp = np.array((mask.cpu().squeeze(axis=0).clamp(0, 1))).transpose((1, 2, 0))
    fill_img = blending(target_temp, output, mask_temp)

    return fill_img

def generate_ca_fill(model,img_path,box,device):
    target_img = trans(Image.open(img_path))
    mask = generate_mask(target_img, box)  # (3,112,112) ,
    ca_mask = 1. - mask[0].unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    target_img = target_img.unsqueeze(0).to(device)
    masked_img = target_img * mask
    source_img = target_img * mask - (1 - mask)
    # inpaint_output = imodel.infer(source_img, mask)
    ca_mask = ca_mask.unsqueeze(dim=0)
    # print(masked_img.size())
    # print(ca_mask.size())
    x1, x2, _ = model(masked_img, ca_mask)
    output = x2 * ca_mask + source_img * (1. - ca_mask)
    output = (output + 1) / 2
    output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
    return output


def generate_ca_inpainting_swap(imodel, smodel, img_path, box, device):
    target_img = trans(Image.open(img_path))
    mask = generate_mask(target_img, box)  # (3,112,112) ,
    ca_mask = 1. - mask[0].unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    target_img = target_img.unsqueeze(0).to(device)
    masked_img = target_img * mask
    source_img = target_img * mask - (1 - mask)
    # inpaint_output = imodel.infer(source_img, mask)
    ca_mask = ca_mask.unsqueeze(dim=0)
    # print(masked_img.size())
    # print(ca_mask.size())
    # masked_img = (masked_img + 1) / 2
    # masked_img = transforms.ToPILImage()(masked_img.cpu().squeeze().clamp(0, 1))
    # masked_img.save('data_occ/masked_whn/a.jpg')
    x1, x2, _ = imodel(masked_img, ca_mask)
    inpaint_output = x2 * ca_mask + masked_img * (1. - ca_mask)
    # inpaint_output = (inpaint_output + 1) / 2
    # inpaint_output = transforms.ToPILImage()(inpaint_output.cpu().squeeze().clamp(0, 1))
    # inpaint_output.save('data_occ/masked_whn/b.jpg')
    with torch.no_grad():
        output, z_id, output_z_id, feature_map, output_feature_map = smodel.forward(inpaint_output, source_img)
    output = (output + 1) / 2
    output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
    target_temp = (target_img + 1) / 2
    target_temp = transforms.ToPILImage()(target_temp.cpu().squeeze().clamp(0, 1))
    mask_temp = np.array((mask.cpu().squeeze(axis=0).clamp(0, 1))).transpose((2, 1, 0))
    fill_img = blending(target_temp, output, mask_temp)

    return fill_img

def generate_mask_ca_inpainting_swap(imodel, smodel, img_path, mask_path, device):
    target_img = trans(Image.open(img_path))
    # masked_img = trans(Image.open(masked_img_path))
    mask = np.asarray(Image.open(mask_path).convert('RGB'))/255.# [3, 112, 112]
    mask = np.where(mask > 0, 1., 0.)  # 112, 112, 3
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (7, 7),iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (7,7), iterations=5)
    # cv2.imshow('',mask)
    mask = np.transpose(mask, (2, 0, 1))
    # print(mask.dtype)
    mask = torch.from_numpy(mask).float()
    # mask = transforms.ToTensor()(mask)
    # mask = mask.float()
    # # print(torch.where(mask < 0.))
    # mask = torch.where(mask > 0., 1., 0.)
    # # print(mask.size())
    ca_mask = mask[0].unsqueeze(0).to(device)
    mask = ca_mask.repeat(3,1,1)

    mask = mask.unsqueeze(0).to(device)
    # # # # # print(mask.size())
    target_img = target_img.unsqueeze(0).to(device)
    masked_img = target_img * (1.-mask)
    source_img = target_img * (1.-mask) - mask
    ca_mask = ca_mask.unsqueeze(dim=0)
    # masked_img = (masked_img + 1) / 2
    # masked_img = transforms.ToPILImage()(masked_img.cpu().squeeze().clamp(0, 1))
    # masked_img.save('data_occ/CFP-FP/a.jpg')
    # # # #
    # # print(masked_img.size())
    # # print(ca_mask.size())
    x1, x2, _ = imodel(masked_img, ca_mask)
    inpaint_output = x2 * ca_mask + masked_img * (1. - ca_mask)

    with torch.no_grad():
        output, z_id, output_z_id, feature_map, output_feature_map = smodel.forward(inpaint_output, source_img)
    output = (output + 1) / 2
    output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
    target_temp = (target_img + 1) / 2
    target_temp = transforms.ToPILImage()(target_temp.cpu().squeeze().clamp(0, 1))
    mask_temp = np.array((mask.cpu().squeeze(axis=0).clamp(0, 1))).transpose((2, 1, 0))
    fill_img = blending(target_temp, output, mask_temp)

    return fill_img




parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, default="./chkpt/vgg220531_last_10.ckpt",
                    help="path of Fusor-net pre-trained file")
parser.add_argument("--gpu_num", type=int, default=0, help="number of gpu")
parser.add_argument("--root", type=str, default='data/LFWs/20220712/', help="root dir of save path")
parser.add_argument("--unocc_dir", type=str, default='data/LFWs/lfw_112/', help="root dir of save path")
parser.add_argument("--occlist",type=str,default='data/occluder/occluders/occluder.txt',help = '')
parser.add_argument("--occluder_dir",type=str,default='data/occluder/occluders/')
parser.add_argument("-m","--mask_type", type=str, default='twoeyes',
                    help="['left_eye', 'right_eye', 'twoeyes', 'nose', 'mouth', 'down_face', 'up_face', 'left_face', 'right_face','random_block','random_part', 'center_block','RMFD']")
parser.add_argument("-f", "--fill_type", type=str, default='ca_id_fill',
                    help="['occ','mask','white','black','gray','ca_fill','ca_id_fill','ca_id_fill_mask','id_fill']")
# python prepare_inpainted_datasets.py --root data/LFWs/20220712/ --unocc_dir data/LFWs/lfw_112/ -m random_part -f ca_id_fill
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')
hp = OmegaConf.load(args.config)
namesList = ['occ', 'mask']
namesList.append(args.fill_type)
#['occ','mask','white','black','gray','ca_fill','ca_id_fill','ca_id_fill_mask','id_fill']

# load ca_inpainter
if namesList.count('ca_fill') or namesList.count('ca_id_fill') or namesList.count('ca_id_fill_mask'):
    cfg = get_config(args.config)
    ca_model = create_ca_model(cfg)

if namesList.count('id_fill') or namesList.count('ca_id_fill') or namesList.count('ca_id_fill_mask'):
    # load id_inpainter
    id_model = Fusor_Net.load_from_checkpoint(args.checkpoint_path, hp=hp)
    id_model.eval()
    id_model.freeze()
    id_model.to(device)
#
trans = gettransform()
dir_unocc = args.unocc_dir
create_dir(args.root)
createDirs(args.root,args.mask_type,namesList)
boxes, names = get_occ_boxs_v2()

for i, subdir in tqdm(enumerate(os.listdir(dir_unocc))):
    for img_name in os.listdir(os.path.join(dir_unocc, subdir)):
        img_path = os.path.join(dir_unocc, subdir, img_name)
        ## CFP-FP & AgeDB
        # img_path = os.path.join(dir_unocc, subdir)
        if args.mask_type:
            box = get_box(args.mask_type)
        for name in namesList:
            if name == 'occ':
                img = PIL_Reader(img_path)
                occlist=occlist_reader(args.occlist)
                res_img = generate_occ(img,occlist,args.occluder_dir,box)
            elif name == 'mask':
                img = trans(Image.open(img_path))
                mask = generate_mask(img,box)
                mask = (mask.numpy()*255).transpose((1, 2, 0)) # -> (H,W,C)
                res_img = Image.fromarray(mask.astype(np.uint8))
            elif name == 'white':
                img = trans(Image.open(img_path))
                mask = generate_mask(img, box)
                res_img = img * mask + (1 - mask)
                res_img = (res_img + 1) / 2
                res_img = transforms.ToPILImage()(res_img.cpu().squeeze().clamp(0, 1))
            elif name == 'black':
                img = trans(Image.open(img_path))
                mask = generate_mask(img, box)
                res_img = img * mask - (1 - mask)
                res_img = (res_img + 1) / 2
                res_img = transforms.ToPILImage()(res_img.cpu().squeeze().clamp(0, 1))
            elif name == 'gray':
                img = trans(Image.open(img_path))
                mask = generate_mask(img, box)
                res_img = img * mask
                res_img = (res_img + 1) / 2
                res_img = transforms.ToPILImage()(res_img.cpu().squeeze().clamp(0, 1))

            elif name == 'ca_fill':
                res_img = generate_ca_fill(ca_model, img_path, box, device)
            elif name == 'id_fill':
                res_img = generate_id_fill(id_model,img_path,box, device)
            elif name == 'ca_id_fill':
                # print(name)
                res_img = generate_ca_inpainting_swap(ca_model, id_model, img_path, box, device)
            elif name == 'ca_id_fill_mask':
                mask_path = 'data_occ/lfw-mask/lfw_mask/'+'/'.join(img_path.split('/')[-2:])
                # print(mask_path)
                res_img = generate_mask_ca_inpainting_swap(ca_model, id_model, img_path, mask_path, device)
                # print(img_path)

            # save result image
            if res_img is not None:
                if args.mask_type:
                    save_name = os.path.join(os.path.join(args.root, args.mask_type + '_' + name), subdir, img_name)
                else:
                    save_name = os.path.join(os.path.join(args.root, name), subdir, img_name)
                create_dir(save_name)
                res_img.save(save_name)

