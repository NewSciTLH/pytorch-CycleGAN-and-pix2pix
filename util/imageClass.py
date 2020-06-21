import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/ericd/pytorch-ssim/')
import pytorch_ssim
import torch
from PIL import Image
#from torchvision import transforms
import pytorch_ssim
import os
import numpy as np
import threading
import concurrent.futures
from pathlib import Path
import matplotlib.pyplot as plt

def focus3(path1, path2):
    img1 = np.array(Image.open(path1).copy())[:,:,:3] #(torch.rand(1, 1, 256, 256))
    img2 = np.array(Image.open(path2).copy()) #(torch.rand(1, 1, 256, 256))
    #print(img1.shape)
    #print(img2.shape)
    mask = np.stack([img2[:,:,3],img2[:,:,3],img2[:,:,3]], axis=2)
    #print(mask.shape)
    Img1 = np.rollaxis(img1*mask, 2) #(torch.rand(1, 1, 256, 256))
    
    Img2 = np.rollaxis(img2[:,:,:3]*mask, 2) #(torch.rand(1, 1, 256, 256))
    #print(Img1.shape)
    #print(Img2.shape)
    pil_to_tensor = torch.from_numpy(Img1).float().unsqueeze_(0)/255
    pil_to_tensor2 = torch.from_numpy(Img2).float().unsqueeze_(0)/255
    v = pytorch_ssim.ssim(pil_to_tensor, pil_to_tensor2).item()
    word = path1.split('/')[-1]
    with open( "all.txt", "a" ) as fout:
        fout.write( f'{word}  {str(v)}\n' )
    
    
def ssimAll(dic):
    path = dic['path']
    A = dic['im']
    out = dic['out']
    path1 = path/'A'/'test'/A
    path2 = path/'B'/'test'/A
    img1 = np.array(Image.open(path1).copy())#(torch.rand(1, 1, 256, 256))
    img2 = np.array(Image.open(path2).copy()) #(torch.rand(1, 1, 256, 256))
    Img1 = np.rollaxis(img1, 2) #(torch.rand(1, 1, 256, 256))
    Img2 = np.rollaxis(img2, 2) #(torch.rand(1, 1, 256, 256))
    pil_to_tensor = torch.from_numpy(Img1).float().unsqueeze_(0)/255
    pil_to_tensor2 = torch.from_numpy(Img2).float().unsqueeze_(0)/255
    v = pytorch_ssim.ssim(pil_to_tensor, pil_to_tensor2).item()    
    with open(out , "a" ) as fout:
            fout.write(f'{A} {str(v)}\n' )
        
def histo(file, fname, density=False):    
    with open(file,'r') as fi:
        fourRGBA = fi.readlines()

    fourRGBAd ={b.split()[0]:float(b.split()[1]) for b in fourRGBA}
    x = list(fourRGBAd.values())
    print(len(x))
    plt.hist(x, density=False, bins=200)  # `density=False` would make counts
    plt.ylabel('# of samples')
    plt.xlabel('Comparison of SSIM values');
    plt.savefig(fname)
    plt.clf()
    

from pylab import *
from scipy.optimize import curve_fit
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
def trimodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)
def fitcurve(new, expected=(.39,.12,1450,.7,.1,50)):
    newd ={b.split()[0]:float(b.split()[1]) for b in new if float(b.split()[1])<1 }
    data = list(newd.values())
    y,x,_ = hist(data,100,alpha=.3,label='data')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    params,cov=curve_fit(bimodal,x,y,expected)
    sigma=sqrt(diag(cov))
    plot(x,bimodal(x,*params),'b')
    legend()
    output ={'params':params, 'sigma':sigma}
    return output    
    
def pltNoBorder(path):
    ax2 = plt.axes([0,0,1,1], frameon=False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    file = Image.open(path)
    ax2.imshow(file)
    plt.pause(.1)
    
def printer2(param, param1, howmany = 40):
    path = '/home/ericd/image_colorization/four/'
    for item in random.sample(list(fourRGBAlocd),howmany):
        path1 = path+'A/train/'+item
        path2 = path+'B/train/'+item
        if param> fourRGBAlocd[item]>param1:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(np.array(Image.open(path1).copy())[:,:,:3])
            axs[0, 0].set_title(f'preprocess no mask ')
            axs[0, 1].imshow(np.array(Image.open(path2).copy())[:,:,:3])
            axs[0, 1].set_title('postprocess no mask ')
            axs[1, 0].imshow(np.array(Image.open(path1).copy()))
            axs[1, 0].set_title('preprocess with mask')
            axs[1, 1].imshow(np.array(Image.open(path2).copy()))
            axs[1, 1].set_title('postprocess with mask')

            
            ax = axs.flat[-2]
            ax.set(xlabel=f'{param}> ssim = {fourRGBAlocd[item]} > {param1}', ylabel='')
                
                
            #for ax in axs.flat[-2:]:
            #    ax.set(xlabel=f'{param}> {fourRGBAd[item]} > {param1}', ylabel='different mask')
            #    break

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()    
    
    
if __name__ == "__main__":

    print('program started it computes ssim of 3 channels after multiplication with mask')
    (_,f,t)=next(os.walk( '/home/ericd/image_colorization/four/lower/A/train'))
    path = '/home/ericd/image_colorization/four/lower/'
    dicT = [{'im':v,'path':'/home/ericd/image_colorization/four/lower/','out':"newfour.txt" } for v in t]
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
        executor.map(ssimAll, dicT) 

    print('program ended')