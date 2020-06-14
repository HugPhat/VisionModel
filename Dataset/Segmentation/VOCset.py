import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

class VOCset(Dataset):
    def __init__(self):
        super(VOCset, self).__init__()

    def __getitem__(self, index):
        return 

    