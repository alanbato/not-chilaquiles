from __future__ import division,print_function
from flask import Flask
app = Flask(__name__)

from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import model_from_json

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

import utils
from utils import plots

import vgg16
from vgg16 import Vgg16

data_api_path = 'data/'
data_finetune_path = 'data/finetune/'

vgg2 = Vgg16()
# Get batches for finetuning
batch_size = 64
batches = vgg.get_batches(data_finetune_path+'train', batch_size=batch_size)
batches.nb_class = batches.num_class
batches.nb_sample = batches.samples
batches.nb_steps = batches.nb_sample // batch_size

vgg2.finetune(batches)
vgg2.model.load_weights('not_chilaquiles_model.h5')
vgg2.compile()

@app.route("/")
def hello():
    return "Hello World!"
@app.route("/food>", methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['photo']
        f.save(data_path + 'api/food_photo.jpg')
        batch = vgg.get_batches(data_path, batch_size=1)
        batch.nb_class = batch.num_class
        batch.nb_sample = batch.samples
        photo = next(batch)
        vgg2.predict(photo)
