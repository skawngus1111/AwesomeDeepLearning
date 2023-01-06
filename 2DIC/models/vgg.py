import sys

import torch
import torch.nn as nn

configurations = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'], # VGG11
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'], # VGG13
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'], # VGG16
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # VGG19
}

class BasicConv(nn.Module) :
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(BasicConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class VGG(nn.Module) :
    def __init__(self, model, num_channels=3, num_classes=1000):
        super(VGG, self).__init__()

        self.in_channels = num_channels
        self.num_classes = num_classes

        if model == 'VGG11' :
            configuration = configurations['A']
        elif model == 'VGG13' :
            configuration = configurations['B']
        elif model == 'VGG16' :
            configuration = configurations['D']
        elif model == 'VGG19' :
            configuration = configurations['E']
        else :
            print('wrong model choice : [VGG11, VGG13, VGG16, VGG19]')
            sys.exit()

        self.feature = self._make_layer(configuration)
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        feature = self.feature(x)
        output = self.fc(feature.reshape(x.size(0), -1))

        return output

    def _make_layer(self, configuration):
        layers = []

        for layer in configuration :
            if type(layer) == int :
                layers.append(BasicConv(in_ch=self.in_channels, out_ch=layer, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
                self.in_channels = layer
            elif layer == 'M' :
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        return nn.Sequential(*layers)

def vgg11() :
    return VGG(model='VGG11')

def vgg13() :
    return VGG(model='VGG13')

def vgg16() :
    return VGG(model='VGG16')

def vgg19() :
    return VGG(model='VGG19')

if __name__=='__main__' :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = vgg19().to(device)
    inp = torch.randn((1, 3, 224, 224)).to(device)

    oup = model(inp)