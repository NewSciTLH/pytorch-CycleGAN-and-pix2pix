# input pair (4 channel image 350 by 300 with a mask, path of folder to save)
#output 3 channel image 512 by 512 with black background that is uploaded
#Eric and Jeremy and Luke 6/20/20
import numpy as np
import os
import sys
import threading
import concurrent.futures
import uuid
from google.cloud import storage
from PIL import Image
from datetime import datetime
import argparse
import skimage
#import pdb

startTime = datetime.now()
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/ericd/storagekey.json" #testing

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def downloadBlob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    if not os.path.exists('ckpt/pix2512'):
        os.makedirs('ckpt/pix2512')
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )



def normalizer(image):
    """Homogenize the images"""
    # numpy matrix
    if np.any(image>2):
        n=image/255
    else:
        n = image
    return n

def multiplier(goodP, key):
    """Given an image with 4 channels, it multiplies the first 3 by the mask, and makes it 512 by 512"""
    #goodP path to image
    #key name
    try:
        imageRot =  Image.open(goodP)
    except Exception as e:
        return str(1)+ str(e)

    imageRot = np.array(imageRot)
    imageMask = imageRot[:,:,3]
    imageRGB   = imageRot[:,:,:3]
    channel     = imageMask
    nAlpha      = normalizer(channel)
    nrgb        = normalizer(imageRGB)
    rgb         = nrgb #*nAlpha[:,:,np.newaxis]
    blank_image = np.zeros((512,512,3), dtype=type(rgb))
    blank_image[:rgb.shape[0],:rgb.shape[1],:] =rgb 
    rgb = blank_image
    sourcePath = 'datasets/A/test/'+key+'.png'
    if not os.path.exists('datasets/A'):
        os.makedirs('datasets/A')
    if not os.path.exists('datasets/A/test'):
        os.makedirs('datasets/A/test')

    im = Image.fromarray(np.uint8((rgb)*255))
    im.save(sourcePath)
    return str(os.path.isfile(sourcePath)) + f' Image {sourcePath} saved ' 




def to3(item, key):
    """Given an image with 4 channels, it multiplies the first 3 by the mask, and makes it 512 by 512"""
    # item = "folder/key/AnImageClassButNotAnExtension"
    err = '/n Image preprocessed correctly'
    if not item:
        return str(1) + ' no image received.'
    folder = item.split('/')[0]  
    temp = 'temp/'
    goodP = temp+key+'.png'
    fP = temp+'f'+key+'.png'
    try:
        downloadBlob(folder, item.replace(folder+'/',''), goodP)
        downloadBlob(folder, item.replace(folder+'/','').replace('processed','final'), fP)
    except Exception as e:
        return str(2) + str(e)
    try:
        err = multiplier(goodP, key) + err  #we make the image 512x512
    except Exception as e:
        return str(3)+str(e)
    return err


        
        
def comultiplier(finalP, premask, key):
    """Given an image with 3 channels, an image with 4 channels, takes the channel and puts it on the 3 channel image"""
    #goodP path to image
    #key name
    try:
        image =  Image.open(finalP)
        preMask = Image.open(premask)
    except Exception as e:
        return str(7)+ str(e)
    imageRooth = np.array(image)
    imageMask = np.array(preMask)
    output = np.zeros((imageRooth.shape[0],imageRooth.shape[1],imageRooth.shape[2]+1))
    mask = imageMask[:,:,3]
    nAlpha      = normalizer(mask)
    nrgb        = normalizer(imageRooth)
    output[:,:,:3]   = nrgb
    output[:,:,3]   = nAlpha
    sourcePath = finalP.replace('_fake','')
    
    im = Image.fromarray(np.uint8((output)*255), 'RGBA')
    im.save(sourcePath)
    return '\n'+str(os.path.isfile(sourcePath)) + f' Image {sourcePath} saved ' 


def start(inputPath, outputFolder, name ='new', exp = 'latest'):
    """Given the path of an image and a folder, downloads the image, preprocess it, and applies a NN, then uploads to the folder"""
    # inputPath = "folder/key/AnImageClassButNotAnExtension"
    # outputFolder = "folder/unknonw/folder/structure/"
    
    err = ''
    if inputPath and outputFolder:
        longList = inputPath        
    else:
        return ' Error 0: An input was missing! '
    key = longList.split('/')[-2]

    try:
        err = to3(longList, key) #preprocess 
    except Exception as e:
        return ' Error 1: '+ str(e) + err

    if os.path.isfile('datasets/A/test/'+key+'.png'):
        os.system(f'python3 -u  test.py --dataroot datasets --name {name} --gpu_ids -1 --epoch {exp} ')#run the nn
        try:
            (_, _, filenames) = next(os.walk(f'results/{name}/test_{exp}/images/'))
            file = f'{key}_fake.png'
            if file in filenames:
                folder = outputFolder.split('/')# now we add an alpha mask to this output
                err = err + comultiplier(f'results/{name}/test_{exp}/images/{file}','temp/'+key+'.png', key)
                
                upload_blob(folder[0], f'results/{name}/test_{exp}/images/{file}'.replace('_fake',''),'/'.join(folder[1:])+'/'+file.replace('_fake',''))
                upload_blob(folder[0], 'temp/'+key+'.png','/'.join(folder[1:])+'/'+file.replace('_fake','_input'))
                upload_blob(folder[0], 'temp/f'+key+'.png','/'.join(folder[1:])+'/'+file.replace('_fake','_ideal'))
                os.remove('datasets/A/test/'+key+'.png')
                os.remove('temp/'+key+'.png')
                os.remove('temp/f'+key+'.png')
                os.remove(f'results/{name}/test_{exp}/images/{file}')
                os.remove(f'results/{name}/test_{exp}/images/{file}'.replace('_fake',''))
                os.remove(f'results/{name}/test_{exp}/images/{file}'.replace('_fake','_real'))                    
            else:
                return f'Error 6: file missing on results/{name}/test_{exp}/images/' 
        except Exception as e:
            return 'Error 5: '+str(err)+str(e)
    else:
        return str(err) + '\n file not processed'
    duration = datetime.now() - startTime
    return ("Completed. Duration was " + str(duration))

if __name__ == '__main__':
    #To update: current weights on index or here
    #print('testing')
    #downloadBlob('model_staging','colorization/pix34wmask/latest_net_D.pth', 'ckpt/pix2512/latest_net_D.pth')
    #downloadBlob('model_staging','colorization/pix34wmask/latest_net_G.pth', 'ckpt/pix2512/latest_net_G.pth')
    #downloadBlob('model_staging','colorization/pix34wmask/test_opt.txt', 'ckpt/pix2512/test_opt.txt')
    #toTest = ['18101']
    #toTest = ['18100']
    exp = 130
    import random
    import pandas as pd
    ImForTrained = '/home/ericd/imagesForTrain.txt'
    ssim = '/home/ericd/sample/pytorch-CycleGAN-and-pix2pix/ssim.txt' 
    with open(ImForTrained,'r') as ift:
        IFT_ = ift.readlines()
        IFT = [x[:-1] for x in  IFT_]
    with open(ssim,'r') as iwp:
        SSIM_ = iwp.readlines()
        SSIM = [x[:-1].split()[0] for x in  SSIM_]
    CorIWP = list(set(SSIM).difference(IFT))
    filssim = [b for b in SSIM_ if b[:-1].split()[0] in CorIWP]  
    print(f'out of {len(filssim)} images')
    data6=pd.DataFrame(filssim)
    data6.columns=['Name'] 
    data6[['item','code','mcreated','minput','d','created','input','g']]=data6.Name.str.split(" ",expand=True) 
    data6[["mcreated", "minput"]] = data6[["mcreated", "minput"]].apply(pd.to_numeric)
    data7 = data6[data6['minput']>.809-.043]
    data8 = data7[data7['minput']<.809+.043]
    index = data8['item'].to_list()
    val = random.sample(index, 100)
    print(f'we selected 100 to test on epoch  {exp}')
    os.system('rm -rf /home/ericd/colorization_images/retrain/new/pytorch-CycleGAN-and-pix2pix/datasets/A/test/.ipynb_checkpoints/')
    for value in val:
        try:
            print(start(f'divvyup_store/{value.replace(".png","")}/processed', f'model_staging/colorization/July/new/{exp}/img',name ='new', exp = exp))
        except Exception as e:
            print(e)
            continue
    import os
    os.system(f'gsutil cp /home/ericd/colorization_images/retrain/new/pytorch-CycleGAN-and-pix2pix/ckpt/new/{exp}* gs://model_staging/colorization/July/new/{exp}/w/')
    print('program finished, it used:') 
    print(val)
    print('test ended')
