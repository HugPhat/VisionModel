import torch 
import torchvision.transforms as Tr
from model import ColorModel
import os 
from PIL import Image
import matplotlib.pyplot as plt 
from time import time
import skimage.io as skio
import numpy as np 


model = ColorModel(img_size= 96, cuda=True)
model.load(r'D:\Files\phat_only\pytorchModels\color_3.pth')
test_path = r"D:\Code\Python\vpl-4-captains-body-tracking\data\test"
tests = [os.path.join(test_path, each) for each in os.listdir(test_path)]
model.eval()

def infe_1():
    
    for im in tests:    
        print(im)
        inp = Image.open(im)
        tic = time()
        out = model.predict(inp)
        tac = (time() -tic)*1000
        print(f'--> time: {tac:.4f}ms')
        print(out)
        plt.text(-1,-5,out)
        plt.imshow(inp)
        plt.show()

def infe_parallel():
    model.eval()
    tic0 = time()
    for it, im in enumerate(tests):    
        
        imgs = []
        tic = time()
        for i in range(10):
            img = Image.open(tests[i+it])
            #img = model.transform(img).unsqueeze(0)
            
            imgs.append(img)
        #x = torch.cat(imgs, dim=0)
        #x = x.type(torch.cuda.FloatTensor).cuda()
        #tac = (time() -tic)*1000
        #print(f'==> cvt time: {tac:.4f}ms')
        #########
        #tic = time()
        out = model.predict_paralell(imgs)
        tac = (time() -tic)*1000
        print(f'--> preds time: {tac:.4f}ms')
        print(out)
        #outp = torch.argmax(out, dim=1).cpu().tolist()
        #print(outp)
        
        #plt.text(-1,-5,out)
        #plt.imshow(inp)
        #plt.show() 
    tac0 = time()
    print(f'>>> total time: {(tac0 -tic0):.4f}ms')
infe_parallel()
#infe_1()