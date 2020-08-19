import os 
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../', '..'))

from Backbone.MobileNetv2.MobileNetv2 import MobileNetv2  
from Dataset import ImageNetLabels as lb
              
model = MobileNetv2(debug = False, n_classes=1000)

model.load_state_dict(torch.load(r"Models\MobileNetv2\mnv2.pth"))

dummy_input = torch.ones((1, 3, 224, 224))

dummy_input = Image.open(r"C:\Users\admin\Downloads\images.jpg")

model.eval()

clss, conf = model.predict(dummy_input)
print(lb.ImageNetLabels[clss.item()], " conf: ", conf)


