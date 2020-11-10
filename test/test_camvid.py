import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__),  '..'))

from Dataset.Segmentation.CamVid import *

mask_src = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\CamVid\trainannot'
img_src = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\CamVid\train'
data = torch.utils.data.DataLoader(CamVid(img_src, mask_src, mode='train', ratio=1),
                                    shuffle=True,
                                   batch_size=3, drop_last=False)
print(len(data))
print(len(default_labels))

inv_norm = transforms.Compose(
    [

        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                             std=[1/0.229, 1/0.224, 1/0.255])
    ]
)

for i, (img, label) in enumerate(data):
    print(img.size())
    print(label.size())

    img = inv_norm(img.squeeze(0)).permute(1, 2, 0)
    label = label.squeeze(0).squeeze(0).numpy()
    print(np.unique(label))
    f, axarr = plt.subplots(ncols=2, figsize=(10, 5))
    axarr[0].imshow(img.numpy())
    axarr[1].imshow(label)
    plt.show()
