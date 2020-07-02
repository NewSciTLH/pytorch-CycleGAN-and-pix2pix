FROM tiangolo/python-machine-learning:cuda9.1-python3.6
LABEL description="Contains private information about newSci.Input is a path to file and a folder to store"
WORKDIR /app
COPY . /app
RUN pip install torch torchvision  &&  pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["run.py"]
