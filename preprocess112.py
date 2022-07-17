import argparse

import PIL
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from retinaface.retinaface import Retinaface
from retinaface.align_trans import get_reference_facial_points, warp_and_crop_face

def crop(imgpath,savepath,retinaface,crop_size=(112, 112)):
    img = PIL.Image.open(imgpath)
    img = img.convert('RGB')
    if img is None:
        print('Open Error!',imgpath)
        return
    b = retinaface.detect_image_crop(img)
    if b is None:
        print('Detection Error!',imgpath)
        return
    # print(b)
    landmarks = [
    [b[5], b[6]],
    [b[7], b[8]],
    [b[9], b[10]],
    [b[11], b[12]],
    [b[13], b[14]]
]
    refrence = get_reference_facial_points(default_square=True)
    img = np.array(img)
    warped_face = warp_and_crop_face(img, landmarks, refrence, crop_size)
    img=Image.fromarray(warped_face)
    img.save(savepath)
    return

def process(root,output_dir,has_subdirs=False):
    retinaface = Retinaface()

    dir_origin_path = root
    dir_save_path = output_dir
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    if has_subdirs:#if has subdirs
        for subdir in os.listdir(root):
            dir_origin_path = os.path.join(root, subdir)
            dir_save_path = os.path.join(output_dir, subdir)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            img_names = os.listdir(dir_origin_path)
            for img_name in tqdm(img_names):
                if img_name.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path = os.path.join(dir_origin_path, img_name)
                    save_path = os.path.join(dir_save_path, img_name)
                    crop(image_path,save_path,retinaface,crop_size=(112,112))

    else:
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                save_path = os.path.join(dir_save_path, img_name)
                crop(image_path,save_path,retinaface,crop_size=(112,112))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--has_subdirs", action="store_true")

    args = parser.parse_args()
    process(args.root,args.output_dir,args.has_subdirs)
    # python preprocess112.py --root ./data/celeba/raw/ --output_dir ./data/celeba/celeba112 --has_subdirs
