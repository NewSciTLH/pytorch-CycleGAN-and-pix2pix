# input pair (4 channel image 350 by 300 with a mask, path of folder to save)
#output 3 channel image 512 by 512 with black background that is uploaded
#Eric and Jeremy and Luke 6/7
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
from util.imageClass import focus3
from util.datagen import getPairsData
storage_client = storage.Client.from_service_account_json("/home/ericd/storagekey.json")

startTime = datetime.now()
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/ericd/storagekey.json" #testing


def downloadBlob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    
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
    rgb         = nrgb*nAlpha[:,:,np.newaxis]
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
    goodP2 = temp+key+'f.png'
    try:
        downloadBlob(folder, item.replace(folder+'/',''), goodP)
        downloadBlob(folder, item.replace(folder+'/','').replace('rotcor_crop_of_subject','final'), goodP2)
    except Exception as e:
        return str(2) + str(e)
    try:
        err = multiplier(goodP, key) + err 
        #os.remove(goodP)
    except Exception as e:
        return str(3)+str(e)
    return err


def blur_edges2(inpt,k=5,c=6):
    """post process to remove black border on the head"""
    img=np.copy(inpt)
    edges=skimage.filters.laplace(img[:,:,3],k)
    for edge in np.argwhere(edges!=0):
        img[edge[0],edge[1],0:3]=np.array([0,0,0])
    new_edges=np.copy(edges)
    counter=0
    while True:
        index=np.argwhere(new_edges!=0)
        counter+=1
        new_img=np.copy(img)
        for edge in index:
            img_mask=img[edge[0]-1:edge[0]+2,edge[1]-1:edge[1]+2,0:3]
            edge_mask=new_edges[edge[0]-1:edge[0]+2,edge[1]-1:edge[1]+2]
            temp=np.zeros(edge_mask.shape)
            mask=np.logical_and(edge_mask==0,np.sum(img_mask,axis=2)!=0.0)
            temp[mask]=1.0
            mask=np.logical_and(edge_mask!=0,np.sum(img_mask,axis=2)!=0.0)
            temp[mask]=1.0
            temp[np.sum(img_mask,axis=2)<np.sum(img_mask[1,1])]=0.0
            img_mask=img_mask*np.repeat(np.expand_dims(temp,2),3,axis=2)
            if np.sum(temp)!=0: 
                new_img[edge[0],edge[1],0:3]=np.sum(np.sum(img_mask,axis=0),axis=0)/np.sum(temp)
                edges[edge[0],edge[1]]=0
            else: 
                continue;
        new_edges=edges
        if counter>c:
            return new_img
        img=new_img
        
        
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
    output = blur_edges2(output)
    sourcePath = finalP.replace('_fake','')
    
    im = Image.fromarray(np.uint8((output)*255), 'RGBA')
    im.save(sourcePath)
    return '\n'+str(os.path.isfile(sourcePath)) + f' Image {sourcePath} saved ' 


def start(inputPath):
    """Given the path of an image and a folder, downloads the image, preprocess it, and applies a NN, then uploads to the folder"""
    # inputPath = "folder/key/AnImageClassButNotAnExtension"
    # outputFolder = "folder/unknonw/folder/structure/"
    err = ''
    if inputPath:
        longList = inputPath
        
    else:
        return ' Error 0: An input was missing! '
    key = longList.split('/')[-2]
    #assert len(longList.split('/')) == 3 #we assume the input comes from reconciliation

    try:
        err = to3(longList, key) #preprocess 
    except Exception as e:
        return ' Error 1: '+ str(e) + err

def finalizer(inputPath):  
    err = ''
    if inputPath:
        longList = inputPath
        
    else:
        return ' Error 0: An input was missing! '
    key = longList.split('/')[-2]
    try:
        print('ml applied')
        (_, _, filenames) = next(os.walk('results/pix2512/test_latest/images/'))
        file = f'{key}_fake.png'
        if file in filenames:
            #folder = outputFolder.split('/')# now we add an alpha mask to this output
            print('we call comultiplier')
            err = err + comultiplier(f'results/pix2512/test_latest/images/{file}','temp/'+key+'.png', key)

            created =  f'results/pix2512/test_latest/images/{file}'.replace('_fake','')
            ideal = f'temp/{key}f.png'
            print('we call focus3')
            focus3(created,ideal)
            #upload_blob(folder[0], f'results/pix2512/test_latest/images/{file}'.replace('_fake',''),'/'.join(folder[1:])+'/'+file.replace('_fake',''))
            #divvyup_store/368194/final
            #          \
            print('removing')
            os.remove('datasets/A/test/'+key+'.png')
            os.remove('temp/'+key+'.png')
            os.remove('temp/'+key+'f.png')
            os.remove(f'results/pix2512/test_latest/images/{file}')
            os.remove(f'results/pix2512/test_latest/images/{file}'.replace('_fake',''))
            os.remove(f'results/pix2512/test_latest/images/{file}'.replace('_fake','_real'))
        else:
            print( f'{key}_fake.png missing')
    except Exception as e:
        return 'Error 5: '+str(err)+str(e)
    
if __name__ == '__main__':
    #6/7 there is a change on the structure of input files from 
    #divvyup_store/photoID/...
    #to
    #divvyup_store/productType/photoID/...
    print('testing')
    #downloadBlob('model_staging','colorization/pix2pix/pix2512/latest_net_D.pth', 'ckpt/pix2512/latest_net_D.pth')
    #downloadBlob('model_staging','colorization/pix2pix/pix2512/latest_net_G.pth', 'ckpt/pix2512/latest_net_G.pth')
    #downloadBlob('model_staging','colorization/pix2pix/pix2512/loss_log.txt', 'ckpt/pix2512/loss_log.txt')
    #downloadBlob('model_staging','colorization/pix2pix/pix2512/test_opt.txt', 'ckpt/pix2512/test_opt.txt')
    #divvyup_store/368194/rotcor_crop_of_subject divvyup_store/368194/final
    Q = ''' SELECT  rotcor_crop_of_subject
    FROM  divvyup_metadata.metadata
    WHERE mask_quality IN ('good','okay') AND rotcor_crop_of_subject IS NOT NULL
    '''
    total =  getPairsData(Q, 'rotcor_crop_of_subject')
    size = 25
    for i in range(int(len(total)/size)+1):
        toDo =  total[i:i+size]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            executor.map(start,toDo)
        
        os.system(f'python3 -u  test.py --dataroot datasets   --num_test {len(toDo)}')#run the nn
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            executor.map(finalizer,toDo)
    
    duration = datetime.now() - startTime
    print("Completed. Duration was " + str(duration))
    
    
    