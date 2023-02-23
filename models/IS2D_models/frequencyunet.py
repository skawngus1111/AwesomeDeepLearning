import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

import numpy as np

class FrequencyUNet(nn.Module) :
    def __init__(self, image_size, in_dim, n_class, num_filters, spectrum_ratio=0.1):
        super(FrequencyUNet, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()

        self.image_size = image_size
        self.in_dim = in_dim
        self.n_class = n_class
        self.num_filters = num_filters
        self.spectrum_ratio = spectrum_ratio

        act_fn = nn.LeakyReLU(0.2, inplace=True)

        # Encoding Parts
        self.down1 = conv_block_2(self.in_dim, self.num_filters, act_fn)
        self.pool1 = maxpool()
        self.down2 = conv_block_2(self.num_filters * 1, self.num_filters * 2, act_fn)
        self.pool2 = maxpool()
        self.down3 = conv_block_2(self.num_filters * 2, self.num_filters * 4, act_fn)
        self.pool3 = maxpool()
        self.down4 = conv_block_2(self.num_filters * 4, self.num_filters * 8, act_fn)
        self.pool4 = maxpool()

        self.bridge = conv_block_2(self.num_filters * 8, self.num_filters * 16, act_fn)

        # Decoding Parts
        self.trans1 = conv_trans_block(self.num_filters * 16, self.num_filters * 8, act_fn)
        self.up1    = conv_block_2(self.num_filters * 16, self.num_filters * 8, act_fn)
        self.trans2 = conv_trans_block(self.num_filters * 8, self.num_filters * 4, act_fn)
        self.up2    = conv_block_2(self.num_filters * 8, self.num_filters * 4, act_fn)
        self.trans3 = conv_trans_block(self.num_filters * 4, self.num_filters * 2, act_fn)
        self.up3    = conv_block_2(self.num_filters * 4, self.num_filters * 2, act_fn)
        self.trans4 = conv_trans_block(self.num_filters * 2, self.num_filters * 1, act_fn)
        self.up4    = conv_block_2(self.num_filters * 2, self.num_filters * 1, act_fn)

        # output block
        self.out = nn.Sequential(
            nn.Conv2d(self.num_filters, self.n_class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        x_range = np.arange(0, self.image_size) - int(self.image_size / 2)
        y_range = np.arange(0, self.image_size) - int(self.image_size / 2)

        x_ms, y_ms = np.meshgrid(x_range, y_range)
        self.R = np.sqrt(x_ms ** 2 + y_ms ** 2)

    def forward(self, x):
        x_fft = fft.fftshift(fft.fft2(x))
        mean_spectrum = torch.mean(torch.abs(x_fft), dim=0)
        total_spectrum = torch.sum(mean_spectrum, dim=(0, 1, 2))

        for radius in range(0, self.image_size//2):
            idxx, idxy = np.where(self.R <= radius)
            partial_spectrum = torch.sum(mean_spectrum[:, idxx, idxy], dim=(0, 1))
            current_ratio = partial_spectrum / total_spectrum
            if self.spectrum_ratio <= current_ratio: break

        low_freq_mask = torch.zeros((self.image_size, self.image_size))
        low_freq_mask[idxx, idxy] = 1

        x = torch.abs(fft.ifft2(fft.ifftshift(x_fft * low_freq_mask.to(x.get_device()))))

        # feature encoding
        down1 = self.down1(x)
        pool1 = self.pool1(down1)
        down2 = self.down2(pool1)
        pool2 = self.pool2(down2)
        down3 = self.down3(pool2)
        pool3 = self.pool3(down3)
        down4 = self.down4(pool3)
        pool4 = self.pool4(down4)

        bridge = self.bridge(pool4)

        # feature decoding
        trans1  = self.trans1(bridge)
        if trans1.shape[2:] != down4.shape[2:] :
            trans1 = F.interpolate(trans1, down4.shape[2:])
        concat1 = torch.cat([trans1, down4], dim=1)
        up1     = self.up1(concat1)
        trans2  = self.trans2(up1)
        if trans2.shape[2:] != down3.shape[2:] :
            trans2 = F.interpolate(trans2, down3.shape[2:])
        concat2 = torch.cat([trans2, down3], dim=1)
        up2     = self.up2(concat2)
        trans3  = self.trans3(up2)
        if trans3.shape[2:] != down2.shape[2:] :
            trans3 = F.interpolate(trans3, down2.shape[2:])
        concat3 = torch.cat([trans3, down2], dim=1)
        up3     = self.up3(concat3)
        trans4  = self.trans4(up3)
        if trans4.shape[2:] != down1.shape[2:] :
            trans4 = F.interpolate(trans4, down1.shape[2:])
        concat4 = torch.cat([trans4, down1], dim=1)
        up4     = self.up4(concat4)

        out = self.out(up4)

        return out

    def _calculate_criterion(self, y_pred, y_true):
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        return loss

def conv_block_2(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim)
    )

    return model

def conv_trans_block(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        nn.BatchNorm2d(out_dim), act_fn
    )

    return model

def maxpool() :
    pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    return pool

def conv_block(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim),
        act_fn
    )

    return model