import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../', '..'))


from Backbone.Dense.Tiramisu import Tiramisu

dense = Tiramisu(
                    growth_rate=7, 
                    flow=[ 2, 3, 4, 5, 7],
                    mid = 7,
                    num_classes=2,
                    init_weight = True,
                    )

#dense.cuda()

dense.eval()

x = torch.randn([1, 3, 224, 224])

yhat = dense(x)
print(yhat.size())
