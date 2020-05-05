# input pair (4 channel image 350 by 300 with a mask, path of folder to save)
#output 3 channel image 512 by 512 with black background that is uploaded
#Eric and Jeremy 5/5
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


startTime = datetime.now()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"

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
    """Given an image with 4 channels, it multiplies the first 3 by the mask, and makes it 512 by 512#"""
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



def to3(item):
    """Given an image with 4 channels, it multiplies the first 3 by the mask, and makes it 512 by 512"""
    # item = "folder/key/AnImageClassButNotAnExtension"
    err = '/n Image preprocessed correctly'
    if not item:
        return str(1) + ' no image received.'
    folder = item.split('/')[0]    
    key = item.split('/')[1]
    temp = 'temp/'
    goodP = temp+key+'.png'
    try:
        downloadBlob(folder, item.replace(folder+'/',''), goodP)
    except Exception as e:
        return str(2) + str(e)
    try:
        err = multiplier(goodP, key) + err
        os.remove(goodP)
    except Exception as e:
        return str(3)+str(e)
    return err

def start(inputPath, outputFolder):
    """Given the path of an image and a folder, downloads the image, preprocess it, and applies a NN, then uploads to the folder"""
    # inputPath = "folder/key/AnImageClassButNotAnExtension"
    # outputFolder = "folder/unknonw/folder/structure/"
    err = ''
    if inputPath and outputFolder:
        longList = inputPath
        
    else:
        return ' Error 0: An input was missing! '
    key = longList.split('/')[1]
    assert len(longList.split('/')) == 3 #we assume the input comes from reconciliation

    try:
        err = to3(longList) #preprocess
    except Exception as e:
        return ' Error 1: '+ str(e) + err

    if os.path.isfile('datasets/A/test/'+longList.split('/')[1]+'.png'):
        os.system(f'python3 -u  test.py --dataroot datasets   --num_test {len(longList)}')#run the nn
        try:
            (_, _, filenames) = next(os.walk('results/pix2512/test_latest/images/'))
            for file in filenames:
                if 'fake' in file and key in file:
                    folder = outputFolder.split('/')
                    upload_blob(folder[0], f'results/pix2512/test_latest/images/{file}','/'.join(folder[1:])+'/'+file)
        except Exception as e:
            return 'Error 5: '+str(err)+str(e)
    else:
        return str(err) + '/n file not processed'
    duration = datetime.now() - startTime
    return ("Completed. Duration was " + str(duration))