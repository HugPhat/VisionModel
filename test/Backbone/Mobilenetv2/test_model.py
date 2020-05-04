import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../', '..'))

from Backbone.MobileNetv2.MobileNetv2 import MobileNetv2                

model = MobileNetv2(debug = True)