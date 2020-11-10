import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 

sys.path.insert(0, os.path.join(os.path.dirname(__file__),  '..'))

from Dataset.Segmentation.VOCseg import VOCseg

mask_src = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\PASCALVOC\VOCdevkit\VOC2012\SegmentationClass'
img_src = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\PASCALVOC\VOCdevkit\VOC2012\JPEGImages'
data = torch.utils.data.DataLoader(VOCseg(img_src, mask_src, mode='val', ratio=0.2), 
batch_size=1, drop_last=False)
print(len(data))



inv_norm = transforms.Compose(
    [

        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                             std=[1/0.229, 1/0.224, 1/0.255])
    ]
)

for i, (img, label) in enumerate(data):
    print(img.size())
    print(label.size())
    
    img = inv_norm(img.squeeze(0)).permute(1,2,0)
    label = label.squeeze(0).squeeze(0).numpy()
    print(np.unique(label))
    f, axarr = plt.subplots(ncols=2, figsize=(10, 5))
    axarr[0].imshow(img.numpy())
    axarr[1].imshow( label)
    plt.show()
