from google.cloud import bigquery
from google.cloud import storage
import threading
import concurrent.futures
from itertools import tee

from pathlib import Path
import os
queryclient = bigquery.Client.from_service_account_json("/home/ericd/bqkey.json")
storage_client = storage.Client.from_service_account_json("/home/ericd/storagekey.json")

#Needed to call bigquery

def downloadBlob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    #storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def getData(QUERY, path):
    results = queryclient.query(QUERY).result()
    print("Number of New Elements = ", results.total_rows)
    output = [{'file':r['mask_of_subjects_face'], 'path':path} for r in results]
    return output

def getPairsData(QUERY, class1):
    results = queryclient.query(QUERY).result()
    print("Number of New Elements = ", results.total_rows)
    output = [r[class1] for r in results]
    return output
    #if avoid:
    #    (_,_,f) = next(os.walk('/home/ericd/image_colorization/four/AB/train'))
    #    f2 = [ 'divvyup_store/'+b[:-4]+'/final' for b in f]
    #    output = list(set(output).difference(set(f2)))

def downloadImage(data):
    dataD = data['file']
    folder  = data['path']
    bucket,  name  = dataD.split('/')[0], dataD.split('/')[1]
    pName =  Path(name)
    pFolder = Path(folder)
    try:
        downloadBlob(bucket, str(pName/'final'), str(pFolder/'B'/'test'/name))
        downloadBlob(bucket, str(pName/'processed'),str(folder/'A'/'test'/name))
    except Exception as e:
        print(str(e))
    
def comb(path='/home/ericd/image_colorization/four/lower/'):
    os.system(f'python3 /home/ericd/pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py --fold_A {path}A --fold_B {path}B --fold_AB {path}AB') 
    print('images combined')
if __name__ == '__main__':
    QUERY = """
    SELECT mask_of_subjects_face
    FROM  divvyup_metadata.metadata
    WHERE mask_quality IN ('good','okay') AND rotcor_crop_of_subject IS NOT NULL AND subject_class  IN ('face')
    LIMIT 2000       
    """

    total = getData(QUERY)
    print(len(total), total[0])
    #for value in total:
    #    downloadImage(value)
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        executor.map(downloadImage, total)   
    print('combining')
    comb()
    print('program ended')    