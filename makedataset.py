#make data
#Eric 5/22
import cv2
import numpy as np
import os
#from IPython.core.debugger import set_trace
import threading
import concurrent.futures
import uuid
from google.cloud import bigquery
from google.cloud import storage
from PIL import Image
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'
def downloadBlob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    #storage_client = storage.Client('project=newsci-1532356874110')
    #storage.Client.from_service_account_json("/divvyup-data/data_wrangling_scripts/servacc.json")
    storage_client = storage.Client.from_service_account_json("key.json")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
def getData():
    queryclient = bigquery.Client.from_service_account_json('bqkey.json')#()
    query="""
    SELECT M.mask_of_subjects_face, M.rotcor_crop_of_subject 
    FROM divvyup_metadata.metadata M JOIN divvyup_metadata.inputNNquality N 
    ON M.rotcor_crop_of_subject = N.rotcor_crop_of_subject
    WHERE 
    (M.mask_of_subjects_face IS NOT NULL
    AND
    M.rotcor_crop_of_subject IS NOT NULL
    AND  N.mask_quality = 'good'  AND M.mask_quality = 'good'  )
    ORDER BY RAND()
    LIMIT 20000
    """

    results = queryclient.query(query).result()
    print(f'Our data has {results.total_rows} total rows')
    longList = [ (item['rotcor_crop_of_subject'], item['mask_of_subjects_face']) for item in results  ]
    print(f'longList created with {len(longList)} items' )
    print(longList[0])
    return longList



def normalizer(image):
    # to homogenize the images
    if np.any(image>2):
        n=image/255
    else:
        n = image
    return n

#we multiply with the channels
def whitening(goodP,key):
    try:
        img =  Image.open(goodP)
    except Exception as e:
        return str(1)+ str(e)
    
    img = np.array(img)
    img = normalizer(img)
    mask = img[:,:,3].copy()
    rgb = img[:,:,:3].copy()
    rgb[np.where(mask < 0.1)] = 1.0
    
    blank_image = np.ones((512,512,3), dtype=type(rgb))
    
    blank_image[:rgb.shape[0],:rgb.shape[1],:] =rgb #* blank_image[:rgb.shape[0],:rgb.shape[1],:] #test with white blackground
    rgb = blank_image
    
    sourcePath = 'datasets/A/test/'+key+'.png'
    
    if not os.path.exists('datasets/A'):
        os.makedirs('datasets/A')
    if not os.path.exists('datasets/A/test'):
        os.makedirs('datasets/A/test')
    #im = Image.fromarray(np.uint8((rgb)*255))
    #im.save(sourcePath)
    
    im = Image.fromarray(np.uint8((rgb)*255))
    im.save(sourcePath)


def multiplier(goodP, badP, key, path):
    try:
        imageMask   = cv2.imread(goodP, cv2.IMREAD_UNCHANGED) #change path
        imageRot    = cv2.imread(badP, cv2.IMREAD_UNCHANGED) #change path
    except Exception as e:
        print(str(e))
        return 1
    
    imageRGB    = imageMask[:,:,:3].copy()
    imageRGB2   = imageRot[:,:,:3].copy()
    if imageRGB.shape != imageRGB2.shape:
        print(f'{key} have bad shape! {imageRGB2.shape}')
        return 0
    
    channel     = imageMask[:,:,3].copy()
    nAlpha      = normalizer(channel)
    nrgb        = normalizer(imageRGB)
    
    #    
    nrgb[np.where(nAlpha < 0.1)] = 1.0
    
    blank_image = np.ones((512,512,3), dtype=type(nrgb))
    
    blank_image[:nrgb.shape[0],:nrgb.shape[1],:] =nrgb #* blank_image[:rgb.shape[0],:rgb.shape[1],:] #test with white blackground
    rgb = blank_image

    #make input
    nrgb2        = normalizer(imageRGB2)
    nrgb2[np.where(nAlpha < 0.1)] = 1.0
    blank_image2 = np.ones((512,512,3), dtype=type(nrgb2))
    blank_image2[:nrgb2.shape[0],:nrgb2.shape[1],:] =nrgb2 #* blank_image[:rgb.shape[0],:rgb.shape[1],:] #test with white blackground
    rgb2 = blank_image2

    sourcePath = path+'A/train/'+key+'.png'
    targetPath = path+'B/train/'+key+'.png'
    
    cv2.imwrite(targetPath, ) #Here we saved it with values from 0 to 1.np.uint8( rgb*255)
    cv2.imwrite(sourcePath, np.uint8( rgb2*255))
    
    #display(targetPath)
    #display(sourcePath)

    if os.path.isfile(targetPath) and os.path.isfile(sourcePath):
        print(f'Image {key} saved ')



def to3(item):
    # (item['rotcor_crop_of_subject'], item['mask_of_subjects_face'])
    # divvyup_store/170427/rotcor_crop_of_subject, divvyup_store/170427/final
    key = item[0].split('/')[1]
    path = '/home/ericd/image_colorization/pix2pixData/pytorch-CycleGAN-and-pix2pix/datasets/pure/'
    temp = path+'temp/'
    # Load Data
    bad = item[0] #['rot']
    good = item[1] #['mask']
    goodP = temp+key+'.png'
    badP=temp+key+'_.png'
    try:
        downloadBlob('divvyup_store', good.replace('divvyup_store/',''), goodP)
        downloadBlob('divvyup_store', bad.replace('divvyup_store/',''), badP)
        multiplier(goodP, badP, key, path)
        os.remove(goodP)
        os.remove(badP)
    except Exception as e:
        print(e)
        return 2

longList = getData()
#for item in longList:
#    to3(item)
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    executor.map(to3,  longList)

print('program combining images')   
os.system('nohup python3 -u /home/ericd/image_colorization/pix2pixData/pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py --fold_A /home/ericd/image_colorization/pix2pixData/pytorch-CycleGAN-and-pix2pix/datasets/pure/A --fold_B /home/ericd/image_colorization/pix2pixData/pytorch-CycleGAN-and-pix2pix/datasets/pure/B --fold_AB /home/ericd/image_colorization/pix2pixData/pytorch-CycleGAN-and-pix2pix/datasets/pure/AB > makedatat.out&')
print('program ended')
