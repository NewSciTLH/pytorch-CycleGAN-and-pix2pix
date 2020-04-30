# input pair (4 channel image 350 by 300 with a mask, path of folder to save)
#output 3 channel image 512 by 512 with black background that is uploaded
#Eric 2/24
#import cv2
import numpy as np
import os
import sys
#from IPython.core.debugger import set_trace
import threading
import concurrent.futures
import uuid
from google.cloud import bigquery
from google.cloud import storage
from PIL import Image
from datetime import datetime
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("inputPath", type=str, help="the path to the input image")
parser.add_argument("outputFolder", type=str, help="the path to the output folder")
args = parser.parse_args()


startTime = datetime.now()
os.system('export GOOGLE_APPLICATION_CREDENTIALS="id.json"')

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
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
def getData(): # predicted_mask have the mask while rotcrop has the image
    queryclient = bigquery.Client()
    query="""
    SELECT M.predicted_mask 
    FROM divvyup_metadata.metadata M JOIN divvyup_metadata.inputNNquality N 
    ON M.rotcor_crop_of_subject = N.rotcor_crop_of_subject
    WHERE 
    (M.predicted_mask IS NOT NULL
    AND
    M.rotcor_crop_of_subject IS NOT NULL
    )
    ORDER BY RAND()
    LIMIT 100
    """

    results = queryclient.query(query).result()
    print(f'Our data has {results.total_rows} total rows')
    longList = [  item['predicted_mask'] for item in results  ]
    print(f'longList created with {len(longList)} items' )
    
    return longList



def normalizer(image):
    # to homogenize the images
    if np.any(image>2):
        n=image/255
    else:
        n = image
    return n

#we multiply with the channels
def multiplier(goodP, key, path):
    #We input a rgb(a) image and a mask file
    try:
        imageRot =  Image.open(goodP)
        #imageMask   =  np.load(badP ) #this should be read with numpy from now on
        #imageRot    = cv2.imread(goodP, cv2.IMREAD_UNCHANGED) #change path
    except Exception as e:
        print('error 1')
        print(str(e))
        return 1
    imageRot = np.array(imageRot)
    imageMask = imageRot[:,:,3]
    #imageRGB    = imageMask[:,:,:3]
    imageRGB   = imageRot[:,:,:3]
    print( imageMask.shape, imageRGB.shape)

    #if imageRGB.shape != imageRGB2.shape:
    #    print(f'{key} have bad shape! {imageRGB2.shape}')
    #    return 0
    
    channel     = imageMask
    nAlpha      = normalizer(channel)

    nrgb        = normalizer(imageRGB)
    rgb         = nrgb*nAlpha[:,:,np.newaxis]
    blank_image = np.zeros((512,512,3), dtype=type(rgb))#here the datatype may be a problem, double?
    blank_image[:rgb.shape[0],:rgb.shape[1],:] =rgb 
    rgb = blank_image

    #make input
    """
    nrgb2       = normalizer(imageRGB2)    
    rgb2        = nrgb2*nAlpha[:,:,np.newaxis]
    blank_image2 = np.zeros((512,512,3), dtype=type(rgb2))#here the datatype may be a problem, double?
    blank_image2[:rgb2.shape[0],:rgb2.shape[1],:] =rgb2 
    rgb2 = blank_image2
    """
    sourcePath = 'datasets/A/test/'+key+'.png'
    #targetPath = path+'B/train/'+key+'.png'
    im = Image.fromarray(np.uint8((rgb)*255))
    im.save(sourcePath)
    #cv2.imwrite(sourcePath, np.uint8( rgb*255)) #Here we saved it with values from 0 to 1.
    #cv2.imwrite(sourcePath, np.uint8( rgb2*255))
    #display(targetPath)
    #display(sourcePath)

    #if os.path.isfile(targetPath) and os.path.isfile(sourcePath):
    if os.path.isfile(sourcePath): 
        print(f'Image {sourcePath} saved ')
    else: 
        print('error 3')



def to3(item):
    # (item['rotcor_crop_of_subject'], item['mask_of_subjects_face'])
    # divvyup_store/170427/rotcor_crop_of_subject, divvyup_store/170427/final
    key = item.split('/')[1]
    path = 'imgs/'
    #path = '/home/ericd/image_colorization/corrected/datasets/final/'
    temp = path+'temp/'
    # Load Data
    good = item #['img']
    #bad = item[1] #['mask']
    goodP = temp+key+'.png'
    #badP=temp+key+'.npy'
    try:
        downloadBlob('divvyup_store', good.replace('divvyup_store/',''), goodP)
        print(goodP)
        #downloadBlob('divvyup_store', bad.replace('divvyup_store/',''), badP)
        multiplier(goodP, key, path)
        os.remove(goodP)
        #os.remove(badP)
    except Exception as e:
        print('error 2')
        print(e)
        return 2

if args.inputPath and args.outputFolder:
    longList = args.inputPath
else:
    longList = []
    print('No input image given!!!')
    #longList = getData()

#assert len(longList[0].split('.')) == 1 
# In case we received several request per minute,  we can query them keep the line below
to3(longList)
#with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
#    executor.map(to3,  longList)

if os.path.isfile( 'datasets/A/test/'+longList.split('/')[1]+'.png'):
    print(f'starting improving ilumination.  {datetime.now()-startTime} we preprocessed the image')   
    os.system(f'python3 -u  test.py --dataroot datasets --checkpoints_dir ckpt  --num_test {len(longList)}')
    try:
        (_, _, filenames) = next(os.walk('results/pix2512/test_latest/images/'))
        print(f'len(filenames) file created')
        for file in filenames:
            if 'fake' in file:
                folder = args.outputFolder.split('/')
                upload_blob(folder[0], f'results/pix2512/test_latest/images/{file}','/'.join(folder[1:])+'/'+file)
    except Exception as e:
        print('error 5')
        print(e)


print(f'total time {datetime.now()-startTime}')
#os.system('rm -f "/home/ericd/tests/Dockerpix/docs/datasets/A/test/*"')
print('program ended')



