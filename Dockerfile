FROM tiangolo/python-machine-learning:cuda9.1-python3.6
LABEL description="Contains private information about newSci.Input is a path to file and a folder to store"
COPY ./pytorch-CycleGAN-and-pix2pix .
RUN pip install torch torchvision  &&  pip install -r requirements.txt
CMD  python3 -u  test.py --dataroot datasets --checkpoints_dir /checkpoints  --load_size 512   --phase test > test.out 
