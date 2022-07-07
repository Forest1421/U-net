import os

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import writer, SummaryWriter
from test import *
from data import *
from net import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().cuda()

weight_path = 'params/unet.pth'
train_data_path = r'D:\train_data'
val_data_path = r'D:\val_data'
save_path = 'train_image'
_input1 = r'D:\U_net\test_dataset\1.jpg'
_input2 = r'D:\U_net\test_dataset\2.jpg'

if __name__ == '__main__':
    train_data_loader = DataLoader(MyDataset(train_data_path), batch_size=4, shuffle=True)
    val_data_loader = DataLoader(MyDataset(val_data_path), batch_size=4, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()
    writer = SummaryWriter("trainloss_figure")
    epoch = 1

    while True:
        each_epoch_train_loss = 0
        each_epoch_val_loss = 0
        for i, (image, segment_image) in enumerate(train_data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            each_epoch_train_loss += train_loss.item()
            if i % 5 == 0:
                print(f'epoch:{epoch} -- i:{i} -- train_loss:{train_loss.item()}')

                _image = image[0]
                _segment_image = segment_image[0]
                _out_image = out_image[0]

                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                save_image(img, f'{save_path}/{i}.png')

        for i, (image, segment_image) in enumerate(val_data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            each_epoch_val_loss += train_loss.item()

        each_epoch_train_loss /= 653
        each_epoch_val_loss /= 75
        torch.save(net.state_dict(), weight_path)
        writer.add_scalars('loss', {"train_loss": each_epoch_train_loss,
                                    "val_loss": each_epoch_val_loss}, global_step=epoch)

        print(f'epoch:{epoch} -- train_loss:{each_epoch_train_loss} -- val_loss:{each_epoch_val_loss}')
        image_test(weight_path, _input1, str(epoch) + '_1')
        image_test(weight_path, _input2, str(epoch) + '_2')
        epoch += 1

    writer.close()