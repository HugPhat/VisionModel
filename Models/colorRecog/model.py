import torch
import torch.nn as nn
import torchvision.transforms as transforms

lbs = ['bg', 'blue', 'green', 'purple', 'yellow']

def conv(i, inp, out, k, s, p, use_bn = True, use_do =False):
    m = nn.Sequential()
    m.add_module("b_conv" + str(i), nn.Conv2d(inp, out, kernel_size=k, stride=s, padding=p))
    if use_bn:
        m.add_module("b_bn" + str(i), nn.BatchNorm2d(out))
    
    m.add_module("b_act" + str(i), nn.ReLU6(inplace=True))
    
    if use_do:
        m.add_module("drop" + str(i), nn.Dropout(0.2))
    
    return m
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
class ColorModel(nn.Module):
    def __init__(self, img_size= 96, use_bn=True, use_drop=False, cuda=False):
        super(ColorModel, self).__init__()
        self.transform = transforms.Compose([
                                    #transforms.CenterCrop(img_size),
                                    transforms.Resize((img_size, img_size)),       
                                    transforms.ToTensor(),
                                    transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        self.model = nn.ModuleList()
        self.model.append(conv(0, 3, 16, 3, 1, 1, use_bn=use_bn, use_do=use_drop))
        self.model.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 48x48
        self.model.append(conv(1, 16, 32, 3, 1, 1, use_bn=use_bn, use_do=use_drop))
        self.model.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 24x24
        self.model.append(conv(2, 32, 32, 3, 1, 1, use_bn=use_bn, use_do=use_drop))
        self.model.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 12x12
        self.model.append(conv(3, 32, 16, 1, 1, 0, use_bn=use_bn, use_do=use_drop))# depth wise
        self.model.append(nn.AvgPool2d(8))
        self.model.append(Flatten())
        self.model.append(nn.Linear(16, 32))
        self.model.append(nn.Linear(32, 5))
        self.model.append(nn.LogSoftmax(dim=1))
        
        if cuda:
            self.FloatTensor = torch.cuda.FloatTensor
            self.LongTensor = torch.cuda.LongTensor
            self.cuda()
        else:
            self.FloatTensor = torch.FloatTensor
            self.LongTensor = torch.LongTensor  
        
    def forward(self, x):
        
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x
    
    def load(self, path):
        chp = torch.load(path)
        self.load_state_dict(chp)   
    
    def predict(self, x):  

        inp = self.transform(x).unsqueeze(0).type(self.FloatTensor)

        with torch.no_grad():    
            out = self(inp)
            out = torch.argmax(out).cpu().item()

        return lbs[int(out)]
    
    def predict_paralell(self, x:list):
        tmp = []
        for each in x:
           tmp.append(self.transform(each).unsqueeze(0)) 
        tmp = torch.cat(tmp, dim=0).type(self.FloatTensor)
        with torch.no_grad():    
            out = self(tmp)
            out = torch.argmax(out, dim=1).cpu().tolist()
        lb = [lbs[each] for each in out]
        
        return lb
        
if __name__ == '__main__':
    import os 
    from PIL import Image
    import matplotlib.pyplot as plt 
    from time import time
    
    x = torch.rand([100, 3, 96, 96])
    m = ColorModel(cuda=False)
    m.eval()
    m.cuda()
    x = x.type(torch.cuda.FloatTensor)
    (m(x))
    
    tic = time()
    (m(x))
    tac = (time() -tic)*1000
    print(f'--> time: {tac:.4f}ms')
    #model.load_state_dict(torch.load(r'D:\Files\phat_only\pytorchModels\color.pth'))
    
    s = 0
    for l in m.parameters():
        s += l.cpu().numel()
    print("num parms ", s )