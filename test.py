import os

import torch

from net import *
from utils import keep_image_size_open
from data import *
from torchvision.utils import save_image

net = UNet().cuda()

save_path = r'D:\U_net\test_result'


def image_test(weights, _input, i):
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully')
    else:
        print('no loading')

    img = keep_image_size_open(_input)
    img_data = transform(img).cuda()
    img_data = torch.unsqueeze(img_data, dim=0)
    out = net(img_data)
    save_image(out, f'{save_path}/{i}.png')



