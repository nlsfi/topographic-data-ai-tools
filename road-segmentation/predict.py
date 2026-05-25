import copy
import os
import PIL
import time
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from augmentations import Augmentation
import argparse
from pathlib import Path
import cv2
from model import build_model
from Prediction import pad_image_with_neighbours, num_neighbours, get_neighbour_ids
import geopandas as gpd
from preprocessing import preprocess_additional_input, preprocess_mask_image
from Prediction import large_prediction


parser = argparse.ArgumentParser()

parser.add_argument("--input-path", type=Path, default=None)
parser.add_argument("--additional-channel-path", type=Path, default=None)
parser.add_argument("--output-path", type=Path, default=None)
parser.add_argument("--load-weights-path", type=Path, default=None)
parser.add_argument("--num-classes", type=int, default=2)
parser.add_argument("--img-size", type=int, default=512)
parser.add_argument("--input-channels", type=int, default=4)
parser.add_argument("--use-nir", action="store_true")

args = parser.parse_args()

use_nir = args.use_nir
CLASSES = args.num_classes

img_size = [args.img_size, args.img_size, args.input_channels]
mask_size = args.img_size

weight_path = args.load_weights_path

large_prediction_output_path = args.output_path

path_dtm = args.additional_channel_path

if args.output_path:
    output_path = args.output_path
else:
    output_path = f"outputs/"

input_img1 = tf.keras.layers.Input(img_size, name='input_layer')

model = build_model(img_size)

augm = Augmentation(img_size)

img_augments = [
    augm.random_flip_left_right,
    augm.random_flip_up_down,
    augm.random_rotate,
    augm.resize,
    # augm.random_crop,
]
mask_augments = [
    augm.random_flip_left_right,
    augm.random_flip_up_down,
    augm.random_rotate,
    augm.resize,
    # augm.random_crop,
]

test_img_augments = [
    # augm.crop,
    augm.resize,
]
test_mask_augments = [
    # augm.crop,
    augm.resize,
]

if weight_path:
    print("\nLOADING WEIGHTS", weight_path)
    model.load_weights(weight_path)
    print("WEIGHTS LOADED\n")

print(model.summary())

test_path = args.input_path

path_dtm = args.additional_channel_path

large_prediction(model, test_path, augm, test_img_augments, img_size,
                 additional_channel_path=path_dtm, output_path=output_path)
tf.keras.backend.clear_session()
