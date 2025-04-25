import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DQNet(nn.Module):
    def __init__(self, in_channels=3, size=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv_layer(in_channels, 64),
        )
        self.pool1 = nn.Sequential(
            conv_block(64, [64, 128, 64]),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pool2 = nn.Sequential(
            conv_block(64, [64, 128, 64]),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pool3 = nn.Sequential(
            conv_block(64, [64, 128, 64]),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        d1 = int(2*size)
        d2 = int(size*size/64)
        self.depth_map = nn.Sequential(
            conv_layer(3*64, d1),
            nn.Dropout(0.2),
            conv_layer(d1, d2),
        )
        self.output = nn.Sequential(
            nn.Conv2d(d2, 1, 3, padding=1),
        )
        self.size = size
        #print(size)

    def forward(self, x):
        x = self.conv1(x)
        pool1 = self.pool1(x)
        pool2 = self.pool2(pool1)
        pool3 = self.pool3(pool2)

        map1 = TF.resize(pool1, (self.size, self.size))
        map2 = TF.resize(pool2, (self.size, self.size))
        map3 = TF.resize(pool3, (self.size, self.size))
        summap = torch.cat((map1, map2, map3), 1)
        #summap = map1 + map2 + map3
        #print(summap.size())

        latent_map = self.depth_map(summap)
        output = self.output(latent_map)
        if self.size==64:
            output = TF.resize(output, (32, 32))

        return summap, output

class DQNetclf(nn.Module):
    def __init__(self, out_clf=3):
        super().__init__()
        self.depth_map = nn.Sequential(
            conv_layer(3*64, 128),
            nn.Dropout(0.2),
            conv_layer(128, 64),
        )
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
        )
        self.clf = nn.Sequential(
            nn.Linear(1*32*32, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, out_clf),
        )

    def forward(self, summap):
        latent_map = self.depth_map(summap)
        output = self.output(latent_map)
        output = TF.resize(output, (32, 32))
        output_flat = output.view(-1, 1*32*32)
        output_clf = self.clf(output_flat)
        return output_clf

def conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ELU(),
    )

def conv_block(in_channels, hidden_channels):
    layers = []
    c_in = in_channels
    for h in hidden_channels:
        c_out = h
        layers.append(
            conv_layer(c_in, c_out)
        )
        c_in = h
    return nn.Sequential(*layers)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
