import argparse
from PIL import Image
from omegaconf import OmegaConf

import torch
from torchvision import transforms
import numpy as np
import cv2
from fusor_net import Fusor_Net
import tools.poissonblending as blending
from ca_inpainter.model.camodel import create_ca_model

def generate_mask_ca_inpainting_swap(imodel, smodel,trans, img_path, mask_path, device):
    # print(Image.open(img_path))

    target_img = trans(Image.open(img_path).convert('RGB'))
    # mask = trans(Image.open(mask_path).convert('RGB'))
    mask = np.asarray(Image.open(mask_path).convert('RGB'))/255.# [3, 112, 112]
    mask = np.where(mask > 0, 1., 0.)  # 112, 112, 3
    # # Denoising
    # # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (7, 7),iterations=5)
    # # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (7,7), iterations=5)
    mask = np.transpose(mask, (2, 0, 1))
    mask = torch.from_numpy(mask).float()
    ca_mask = mask[0].unsqueeze(0).to(device)
    #
    mask = ca_mask.repeat(3,1,1)

    mask = mask.unsqueeze(0).to(device)
    target_img = target_img.unsqueeze(0).to(device)
    masked_img = target_img * (1.-mask)
    source_img = target_img * (1.-mask) - mask
    ca_mask = ca_mask.unsqueeze(dim=0)

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
    return target_img,masked_img,source_img,fill_img

def blending(content_img, output, mask):
    '''
    :param content_img: Image type with size(3,256,256)
    :param output: Image type with size(3,256,256)
    :param mask: (3,256,256), with 0 and 1
    :return: Image type with size(3,256,256)
    '''
    obj_img = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    bg_img = cv2.cvtColor(np.array(content_img), cv2.COLOR_RGB2BGR)
    mask = mask * 255
    mask = mask.astype(np.uint8)
    monoMaskImage = cv2.split(mask)[0]
    br = cv2.boundingRect(monoMaskImage)  # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
    out_img = cv2.seamlessClone(obj_img.astype(np.uint8), bg_img.astype(np.uint8), mask.astype(np.uint8), centerOfBR,
                                cv2.NORMAL_CLONE)
    out_img = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

    return out_img

def gettransform():
    transform_list = [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    return transform


def save_image(imglist, path):
    nbr = len(imglist)
    assert nbr > 0, 'number of images is %s,no images ' % nbr
    result = Image.new(imglist[0].mode, (112 * nbr, 112))
    for i in range(0, nbr):
        result.paste(imglist[i], box=(112 * i, 0))
    result.save(path)

# python inpainting.py --checkpoint_path chkpt/vgg220531_last_10.ckpt --image_path output/example/example1.png --mask_path output/example/mask1.png --output_path output/result/result1.png
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="path of aei-net pre-trained file")
parser.add_argument("--image_path", type=str, required=True,
                    help="path of gt face file")
parser.add_argument("--mask_path", type=str, required=True,
                    help="path of mask file")
parser.add_argument("--output_path", type=str, required=True,
                    help="path of output image")
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu")

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

hp = OmegaConf.load(args.config)
ca_model = create_ca_model(hp)
id_model = Fusor_Net.load_from_checkpoint(args.checkpoint_path, hp=hp)
id_model.eval()
id_model.freeze()
id_model.to(device)
trans = gettransform()
gt_img,masked_img,prior_img,res_img = generate_mask_ca_inpainting_swap(ca_model, id_model, trans, args.image_path, args.mask_path, device)

gt_img = (gt_img + 1) / 2
gt_img = transforms.ToPILImage()(gt_img.cpu().squeeze().clamp(0, 1))
masked_img = (masked_img + 1) / 2
masked_img = transforms.ToPILImage()(masked_img.cpu().squeeze().clamp(0, 1))
prior_img = (prior_img + 1) / 2
prior_img = transforms.ToPILImage()(prior_img.cpu().squeeze().clamp(0, 1))
output = (trans(res_img).unsqueeze(0).to(device) + 1) / 2
output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))

save_image([gt_img,masked_img,prior_img,output],args.output_path)



