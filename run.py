from google.cloud import storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"

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
if __name__ == "__main__":
    try:
        downloadBlob('newsci-models','colorization/pix2pix/pix2512/latest_net_D.pth', 'ckpt/pix2512/latest_net_D.pth')
        downloadBlob('newsci-models','colorization/pix2pix/pix2512/latest_net_G.pth', 'ckpt/pix2512/latest_net_G.pth')
        downloadBlob('newsci-models','colorization/pix2pix/pix2512/loss_log.txt', 'ckpt/pix2512/loss_log.txt')
        downloadBlob('newsci-models','colorization/pix2pix/pix2512/test_opt.txt', 'ckpt/pix2512/test_opt.txt')

    except Exception as e:
        print('error 2')
        print(e)
