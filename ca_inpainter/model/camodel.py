
import os
from ..model.networks112 import Generator
import torch
import torch.nn as nn

def create_ca_model(config):
    device_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
    device_ids = list(range(len(device_ids)))
    ca_model = Generator(config['netG'], True, device_ids)
    ca_model.load_state_dict(torch.load(config['netG']['chkpt_path']))
    ca_model = nn.parallel.DataParallel(ca_model, device_ids=device_ids)
    return ca_model

