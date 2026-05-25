import os
import cv2
import PIL
import copy
import time
import random
import numpy as np
import argparse
from PIL import Image
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from augmentations import Augmentation
from model import build_model
from Training import get_data_generator, dice_coef_loss_bce_sigmoid, dice_coef_loss_bce_softmax, f1_score, iou_coef
from preprocessing import preprocess_additional_input, preprocess_mask_image
from Prediction import pad_image_with_neighbours, num_neighbours, get_neighbour_ids


parser = argparse.ArgumentParser()
parser.add_argument("--train-path", required=True, type=Path)
parser.add_argument("--val-path", required=True, type=Path)
parser.add_argument("--additional-channel-path", type=Path, default=None)
parser.add_argument("--load-weights-path", type=Path, default=None)
parser.add_argument("--save-weights-path", type=Path, default=None)
parser.add_argument("--num-classes", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--initial-epoch", type=int, default=0)
parser.add_argument("--img-size", type=int, default=512)
parser.add_argument("--input-channels", type=int, default=4)
parser.add_argument("--use-nir", action="store_true")

args = parser.parse_args()

training = True if not args.predict else False
use_nir = args.use_nir

train_num = args.initial_epoch

BATCH_SIZE = args.batch_size
CLASSES = args.num_classes

img_size = [args.img_size, args.img_size, args.input_channels]
mask_size = args.img_size

weight_path = args.load_weights_path

large_prediction_output_path = args.output_path

train_path = os.path.join(args.train_path, "images")
val_path = os.path.join(args.val_path, "images")
path_dtm = args.additional_channel_path

mask_path = os.path.join(args.train_path, "labels")
mask_val_path = os.path.join(args.val_path, "labels")

TRAIN_LENGTH = os.listdir(mask_path)
VALIDATION_STEPS = os.listdir(mask_val_path)

STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
if args.output_path:
    output_path = args.output_path
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"Model_{BATCH_SIZE}_{CLASSES}_{train_num}"
else:
    output_name = f"Model_{BATCH_SIZE}_{CLASSES}_{train_num}"
    output_path = f"outputs/{output_name}/"

log_filename = 'logs/' + output_name + '.csv'
if args.save_weights_path:
    checkpoint_path = args.save_weights_path
else:
    checkpoint_path = f"checkpoints/{output_name}/"
checkpoint_path_best = f"{checkpoint_path}/{output_name}/best_result.weights.h5"

input_img1 = tf.keras.layers.Input(img_size, name='input_layer')

model = build_model(img_size)

augm = Augmentation(img_size)

img_augments = [
    augm.random_flip_left_right,
    augm.random_flip_up_down,
    augm.random_rotate,
    # augm.random_crop_1024x1024,
    augm.resize,
    # augm.random_crop,
]
mask_augments = [
    augm.random_flip_left_right,
    augm.random_flip_up_down,
    augm.random_rotate,
    # augm.random_crop_1024x1024,
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


os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
checkpoint_dir = os.path.dirname(checkpoint_path)

os.makedirs(os.path.dirname(log_filename), exist_ok=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path + 'checkpoint_{epoch:03d}.weights.h5',
    save_weights_only=True,
    save_best_only=False,
    verbose=1)

cp_callback_best = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_best,
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      verbose=1,
                                                      monitor="val_output_1_5_f1_score",
                                                      mode="max",
                                                      initial_value_threshold=0.6,
                                                      )


tf.random.set_seed(1234)
callbacks = [
    cp_callback,
    # cp_callback_best,
    CSVLogger(log_filename, separator=",", append=True),
]

if weight_path:
    print("\nLOADING WEIGHTS", weight_path)
    model.load_weights(weight_path)
    print("WEIGHTS LOADED\n")


initial_lr = 1e-4
decay_steps = STEPS_PER_EPOCH * 90
warmup_steps = STEPS_PER_EPOCH * 10
warmup_target = float(initial_lr)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_lr, decay_steps, alpha=1e-4, warmup_steps=warmup_steps, warmup_target=warmup_target)

optimizer1 = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer3 = tf.keras.optimizers.Adam(learning_rate=1e-5)

data_generator = get_data_generator([train_path], [mask_path], path_dtm,
                                    augm, img_augments, mask_augments,
                                    img_size, CLASSES, use_nir)
val_data_generator = get_data_generator([val_path], [mask_val_path], path_dtm,
                                    augm, test_img_augments, test_mask_augments,
                                    img_size, CLASSES, use_nir)


data_generator = data_generator.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

val_data_generator = val_data_generator.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


loss1 = dice_coef_loss_bce_sigmoid
loss2 = dice_coef_loss_bce_softmax

model.compile(optimizer=optimizer1, loss=[loss1, loss1, loss1, loss1, loss2, loss2], metrics={
                                                                'output_1_1': [f1_score],
                                                                'output_1_2': [f1_score],
                                                                'output_1_3': [f1_score],
                                                                'output_1_4': [f1_score],
                                                                'output_1_5': [f1_score],
                                                                'output_2_1': [f1_score]
            },
              loss_weights=[0.5, 0.5, 0.75, 0.75, 1.0, 1.0])

print(model.summary())

model.fit(data_generator,
          epochs=100,
          initial_epoch=train_num,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=VALIDATION_STEPS,
          validation_data=val_data_generator,
          callbacks=callbacks,
          batch_size=BATCH_SIZE)


tf.keras.backend.clear_session()
