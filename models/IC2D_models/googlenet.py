import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels_list[0], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                     nn.BatchNorm2d(out_channels_list[0]), nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, out_channels_list[1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                     nn.BatchNorm2d(out_channels_list[1]), nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels_list[1], out_channels_list[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                     nn.BatchNorm2d(out_channels_list[2]), nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, out_channels_list[3], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                     nn.BatchNorm2d(out_channels_list[3]), nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels_list[3], out_channels_list[4], kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                     nn.BatchNorm2d(out_channels_list[4]), nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                     nn.Conv2d(in_channels, out_channels_list[5], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                     nn.BatchNorm2d(out_channels_list[5]), nn.ReLU(inplace=True))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        return out

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, image_size=224, num_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes

        self.stem_conv = nn.Sequential(nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                                       nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                       nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.BatchNorm2d(192), nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

        self.inception_branch_a3 = InceptionModule(in_channels=192, out_channels_list=[64, 96, 128, 16, 32, 32])
        self.inception_branch_b3 = InceptionModule(in_channels=256, out_channels_list=[128, 128, 192, 32, 96, 64])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception_branch_a4 = InceptionModule(in_channels=480, out_channels_list=[192, 96, 208, 16, 48, 64])
        self.inception_branch_b4 = InceptionModule(in_channels=512, out_channels_list=[160, 112, 224, 24, 64, 64])
        self.inception_branch_c4 = InceptionModule(in_channels=512, out_channels_list=[128, 128, 256, 24, 64, 64])
        self.inception_branch_d4 = InceptionModule(in_channels=512, out_channels_list=[112, 114, 288, 32, 64, 64])
        self.inception_branch_e4 = InceptionModule(in_channels=528, out_channels_list=[256, 160, 320, 32, 128, 128])
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception_branch_a5 = InceptionModule(in_channels=832, out_channels_list=[256, 160, 320, 32, 128, 128])
        self.inception_branch_b5 = InceptionModule(in_channels=832, out_channels_list=[384, 192, 384, 48, 128, 128])
        self.avgpool5 = nn.AvgPool2d(7, 1)

        self.fc = nn.Linear(1024, self.num_classes)
        self.aux_fc1 = InceptionAux(512, self.num_classes)
        self.aux_fc2 = InceptionAux(528, self.num_classes)

    def forward(self, x):
        stem_out = self.stem_conv(x)

        out3 = self.inception_branch_a3(stem_out)
        out3 = self.inception_branch_b3(out3)
        out3 = self.maxpool3(out3)

        out4 = self.inception_branch_a4(out3)
        aux_out1 = self.aux_fc1(out4)

        out4 = self.inception_branch_b4(out4)
        out4 = self.inception_branch_c4(out4)
        out4 = self.inception_branch_d4(out4)
        aux_out2 = self.aux_fc2(out4)

        out4 = self.inception_branch_e4(out4)
        out4 = self.maxpool4(out4)

        out5 = self.inception_branch_a5(out4)
        out5 = self.inception_branch_b5(out5)
        out5 = self.avgpool5(out5)

        out5 = out5.view(out5.shape[0], -1)
        out5 = self.fc(out5)


        return out5#, aux_out1, aux_out2

if __name__=="__main__":
    import torch
    model = GoogLeNet().cuda()
    inp = torch.randn((2, 3, 224, 224)).cuda()
    out5, aux_out1, aux_out2 = model(inp)
    print("out5 shape : ", out5.shape)
    print("aux_out1 shape : ", aux_out1.shape)
    print("aux_out2 shape : ", aux_out2.shape)