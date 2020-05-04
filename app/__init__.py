from flask import Flask
from flask_cors import CORS
import os
import makedata

if os.path.isfile('key.json'):
    print('Setting Credentials...')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"
else:
    print('Setting Credentials...')
    with open('key.json', 'w') as f:
        f.write(os.environ['servacckey'])
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "key.json"

print('Downloading weights...')
makedata.downloadBlob('model_staging','colorization/pix2pix/pix2512/latest_net_D.pth', 'ckpt/pix2512/latest_net_D.pth')
makedata.downloadBlob('model_staging','colorization/pix2pix/pix2512/latest_net_G.pth', 'ckpt/pix2512/latest_net_G.pth')
makedata.downloadBlob('model_staging','colorization/pix2pix/pix2512/loss_log.txt', 'ckpt/pix2512/loss_log.txt')
makedata.downloadBlob('model_staging','colorization/pix2pix/pix2512/test_opt.txt', 'ckpt/pix2512/test_opt.txt')

print('Initializing Flask Server...')
app = Flask(__name__)
CORS(app,origins=['*'])

from app import views