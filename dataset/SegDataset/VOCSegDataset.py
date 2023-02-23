import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

import numpy as np
from PIL import Image

class VOCSegDataset(VOCSegmentation):
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.masks[idx])

        if self.transforms is not None:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transforms(image)
            self._set_seed(seed); label = self.transforms(label) * 255
            label[label > 20] = 0

        return image, label

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

if __name__=='__main__':
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((-5, 5), expand=False),
        transforms.ToTensor(),
    ])

    target_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((-5, 5), expand=False),
        transforms.ToTensor(),
    ])

    train_ds = VOCSegDataset('/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/IS2D_dataset/PASCAL VOC/',
                             year='2012', image_set='train', download=True, transforms=image_transforms)
    val_ds = VOCSegDataset('/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/IS2D_dataset/PASCAL VOC/',
                           year='2012', image_set='val', download=True, transforms=image_transforms)

    np.random.seed(0)
    num_classes = 21
    COLORS = np.random.randint(0, 2, size=(num_classes + 1, 3), dtype='uint8')

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    import matplotlib.pyplot as plt

    for img, target in train_loader:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(np.transpose(img[0].cpu().detach().numpy(), (1, 2, 0)))
        ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([])

        ax[1].imshow(target[0].squeeze().cpu().detach().numpy())
        ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

        plt.show()