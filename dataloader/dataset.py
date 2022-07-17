import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from random import randint
import cv2
import torch
from torch.utils.data import Dataset

def random_regular_mask(img):
    """Generates a random regular hole"""
    mask = torch.ones_like(img)
    s = img.size()
    N_mask = random.randint(1, 5)
    limx = s[1] - s[1] / (N_mask + 1)
    limy = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
        mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
    return mask


def center_mask(img):
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = torch.ones_like(img)
    size = img.size()
    x = int(size[1] / 4)
    y = int(size[2] / 4)
    range_x = int(size[1] * 3 / 4)
    range_y = int(size[2] * 3 / 4)
    mask[:, x:range_x, y:range_y] = 0

    return mask


def random_irregular_mask(img):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(img)
    size = img.size()
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 20
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(16, 64)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            thickness = randint(4, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            radius = randint(4, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    for j in range(size[0]):
        mask[j, :, :] = img_mask < 1

    return mask

class Train_Dataset(Dataset):
    def __init__(self, root, transform=None, mask_type=[0, 1, 2]):
        super(Train_Dataset, self).__init__()
        self.root = root
        self.mtype = mask_type
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transform = transform

    def __getitem__(self, index):
        l = len(self.files)
        s_idx = index % l
        if index >= 4 * l:
            f_idx = s_idx

        else:
            f_idx = random.randrange(l)

        f_img = Image.open(self.files[f_idx])
        s_img = Image.open(self.files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)

        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)
        mask = self.load_mask(f_img, index)
        return f_img, s_img, same, mask

    def __len__(self):
        return len(self.files) * 5

    def load_mask(self, img, index):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mtype) - 1)
        mask_type = self.mtype[mask_type_index]

        # center mask
        if mask_type == 0:
            return center_mask(img)

        # random regular mask
        if mask_type == 1:
            return random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return random_irregular_mask(img)


class Val_Dataset(Dataset):
    def __init__(self, root, transform=None, mask_type=[0]):
        super(Val_Dataset, self).__init__()
        self.root = root
        self.mtype = mask_type
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transfrom = transform

    def __getitem__(self, index):
        l = len(self.files)

        f_idx = index // l
        s_idx = index % l

        f_img = Image.open(self.files[f_idx])
        s_img = Image.open(self.files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transfrom is not None:
            f_img = self.transfrom(f_img)
            s_img = self.transfrom(s_img)

        if f_idx == s_idx:
            same = torch.ones(1)
            mask = torch.ones_like(f_img)

        else:
            same = torch.zeros(1)
            mask = self.load_mask(f_img, index)

        return f_img, s_img, same, mask

    def __len__(self):
        return len(self.files) * len(self.files)

    def load_mask(self, img, index):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mtype) - 1)
        mask_type = self.mtype[mask_type_index]

        # center mask
        if mask_type == 0:
            return center_mask(img)

        # random regular mask
        if mask_type == 1:
            return random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return random_irregular_mask(img)
