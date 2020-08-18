import os 
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../', '..'))

from Backbone.MobileNetv2.MobileNetv2 import MobileNetv2                
model = MobileNetv2(debug = True)

dummy_input = torch.ones((1, 3, 224, 224))
model.train()

x = model(dummy_input)
#print(x.size())
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)