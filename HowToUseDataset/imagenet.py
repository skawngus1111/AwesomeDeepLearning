import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data

import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DATA_PATH = DATASET_DIR

train_dir = os.path.join(ROOT_DATA_PATH, 'train')
val_dir = os.path.join(ROOT_DATA_PATH, 'val')
json_file = os.path.join(ROOT_DATA_PATH, 'imagenet_class_index.json')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.ImageFolder(train_dir, transform)
val_dataset = datasets.ImageFolder(val_dir, transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

idx2label = []
cls2label = {}
with open(json_file, "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

print(idx2label)
print(cls2label)

for i, (input, target) in enumerate(val_loader):
    print(input.shape)
    print(idx2label[target[0]])
    plt.imshow(np.transpose(input[0].cpu().detach().numpy(), (1, 2, 0)))
    plt.show()