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
from losses import dice_coef_loss_bce
from metrics import f1_score, iou_coef
from scipy.ndimage import binary_dilation, binary_erosion
from model_unet3plus import unet3plus
from model_roadvecnet import build_model2


PIL.Image.MAX_IMAGE_PIXELS = None
os.environ['PYTHONHASHSEED'] = str(1)
tf.random.set_seed(1234)
random.seed(123)
np.random.seed(123)


def train_generator(img_dirs, mask_dirs, img_augmentations=None, mask_augmentations=None, additional_path=None, additional_imgs: list = None, output2: bool = False):
    list_images = os.listdir(img_dirs[0])
    if additional_path:
        list_dtm = os.listdir(additional_path + additional_imgs[0])
        im_files = [im.split('.')[0] for im in list_images]
        im_end = list_images[0].split('.')[1]
        dtm_files = [im.split('.')[0] for im in list_dtm]
        a_end = list_dtm[0].split('.')[1]
        im_files = list(set(im_files).intersection(dtm_files))
        mask_files = os.listdir(mask_dirs[0])
        mask_files = [m.split('.')[0] for m in mask_files]
        im_files = list(set(im_files).intersection(mask_files))
        list_images = list(im + '.' + im_end for im in im_files)
    list_images.sort()
    img_i = 0
    while True:
        if img_i >= len(list_images) - 1:
            random.shuffle(list_images)
            img_i = 0

        cur_ind = img_i
        img_path = img_dirs[0]
        arr = np.array(Image.open(img_path + list_images[cur_ind]))
        if arr.shape[-1] == 4:
            arr = arr[:, :, :-1]
        if additional_path:
            add_imgs = []
            for a_img in additional_imgs:
                a_arr = np.load(additional_path + a_img + "/" + list_images[cur_ind].split('.')[0] + '.' + a_end)
                a_arr = a_arr[a_arr.files[0]]
                
                a_arr = np.array(a_arr)  # [:,:,0]
                if a_arr.shape != (*a_arr.shape, 1):
                    a_arr = a_arr.reshape((*a_arr.shape, 1))
                
                add_imgs.append(a_arr)
            
            arr = np.concatenate([arr, *add_imgs], axis=-1)
        
        # add augments
        for aug in img_augmentations:
            arr = aug(arr)
        arr = np.array(arr)
        arr[:,:,:-1] = augm.normalize_data(arr[:,:,:-1])
        
        mask = np.array(Image.open(mask_dirs[0] + list_images[cur_ind].split(".")[0]+'.png'))
        
        # check if mask shape has additional information
        mask_shape = mask.shape
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask_arr = mask.reshape((mask_shape[0], mask_shape[1], 1))
        
        # add augmentations
        for aug in mask_augmentations:
            mask_arr = aug(mask_arr)
        
        mask_arr = np.array(mask_arr)
        mask_arr = np.where(mask_arr > 1, 1, 0)
        
        mask_arr = tf.cast(mask_arr, tf.int32)
        mask_arr = tf.one_hot(mask_arr, 2)
        mask_arr = tf.reshape(mask_arr, (img_size[0], img_size[1], 2))
        
        augm.update_randoms()
        if output2:
            gradients = [np.zeros((512, 512, 1))]
            for z in range(1, mask_arr.shape[-1]):
                asd = np.array(mask_arr[:, :, z]).reshape([img_size[0], img_size[1]])
                asd = asd.astype('uint8')
                dilation = binary_dilation(asd, iterations=1)
                erosion = binary_erosion(dilation, iterations=1)
                gradients.append(np.array(np.bitwise_xor(dilation, erosion).reshape(512, 512, 1)))
            gradients[0] = np.where(gradients[1] > 0, 1, 0)
            grads = tf.one_hot(gradients[0], 2)
            grads = np.array(grads).reshape([512, 512, 2])
    
            img_i += 1
            yield (np.array(arr)), (np.array(mask_arr), np.array(grads))  # x_batches, y_batches  # output_dir, output_dir_masks
        else:
            img_i += 1
            yield (np.array(arr)), (np.array(mask_arr))


training = True
load_weights = False

additional_imgs = ['slope']  # aspect, DTM, hillshade, intensiteetti, slope, Roughness
train_num =      0
crop =           'crop'
data_name =      'ATMU_data'
model_name =     'RVN'  # 'RVN', unet3plus'

TRAIN_LENGTH = 2872
VALIDATION_STEPS = 346
BATCH_SIZE = 1

str_train_num = str(train_num).zfill(3)
additional_imgs_str = "_".join(additional_imgs)
weight_path = f"checkpoints/{model_name}_{data_name}_{BATCH_SIZE}_{additional_imgs_str}/checkpoint_{str_train_num}.weights.h5"

img_size = [512, 512, 3 + len(additional_imgs)]

data_path = 'data/training/'
train_path = data_path + "RGB/image/"
test_path = data_path + "RGB/image_test/"
val_path = data_path + "RGB/image_val/"
path_lidar = data_path + f"/lidar_derived/npz/"

mask_path = data_path + "/labels/label_training/"
mask_test_path = data_path + "/labels/label_test/"
mask_val_path = data_path + "/labels/label_val/"


STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

output_name = f"{model_name}_{data_name}_{crop}_{data_path.split('/')[1]}_{BATCH_SIZE}_{additional_imgs_str}"
log_filename = 'logs/' + output_name + '.csv'
checkpoint_path = f"checkpoints/{model_name}_{data_name}_{BATCH_SIZE}_{additional_imgs_str}/"

checkpoint_path_best = "checkpoints/" + output_name + "_best.weights.h5"
print(output_name)
input_img1 = tf.keras.layers.Input(img_size, name='input_1')

if model_name == 'RVN':
    model = build_model2(img_size)
else:
    model = unet3plus(img_size, 2)


augm = Augmentation(img_size)

img_augments = [
    augm.random_flip_left_right,
    augm.random_flip_up_down,
    augm.random_rotate,
    #augm.random_crop,
    # augm.random_crop_1024x1024,
    augm.resize if crop=='resize' else augm.random_crop,
    # augm.random_brightness,
    # augm.random_contrast,
    # augm.normalize_data
]
mask_augments = [
    augm.random_flip_left_right,
    augm.random_flip_up_down,
    augm.random_rotate,
    # augm.random_crop_1024x1024,
    augm.resize if crop=='resize' else augm.random_crop,
    #augm.random_crop,
]

test_img_augments = [
    augm.resize if crop=='resize' else augm.crop,
    # augm.normalize_data
]
test_mask_augments = [
    augm.resize if crop=='resize' else augm.crop,
]

if not os.path.exists(os.path.dirname(checkpoint_path)):
    os.makedirs(os.path.dirname(checkpoint_path))
checkpoint_dir = os.path.dirname(checkpoint_path)

if not os.path.exists(os.path.dirname(log_filename)):
    os.makedirs(os.path.dirname(log_filename))

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path + f'/checkpoint_' + '_{epoch:03d}.weights.h5',
    save_weights_only=True,
    save_best_only=False,
    verbose=1)
cp_callback_best = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_best,
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      verbose=1,
                                                      monitor="val_output_1_iou_coef",
                                                      mode="max",
                                                      initial_value_threshold=0.6,
                                                      )

callbacks = [
    cp_callback,
    CSVLogger(log_filename, separator=",", append=True),
]

if load_weights:
    model.load_weights(weight_path)
    print("WEIGHTS LOADED")

decay_steps = STEPS_PER_EPOCH * 80
warmup_steps = STEPS_PER_EPOCH * 20
warmup_target = 1e-3
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    1e-12, decay_steps, alpha=1e-7, warmup_steps=warmup_steps, warmup_target=warmup_target)

optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-5)

loss = dice_coef_loss_bce

print(img_size)
if model_name == "RVN":
    output_signature = (
        (tf.TensorSpec(shape=img_size, dtype=tf.float32)),
        (tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32),
         tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32))
    )
    _metrics = {'output_1': [iou_coef, f1_score], 'output_2': [iou_coef, f1_score]}
    _loss = [loss, loss]
else:
    output_signature = (
        (tf.TensorSpec(shape=img_size, dtype=tf.float32)),
        (tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32))
    )
    _metrics = [iou_coef, f1_score]
    _loss = loss


data_generator = tf.data.Dataset.from_generator(lambda:
                                                map(tuple, train_generator([train_path], [mask_path], img_augments,
                                                                           mask_augments, additional_path=path_lidar, additional_imgs=additional_imgs, output2=True if model_name == "RVN" else False)),
                                                output_signature=output_signature
                                                )


val_data_generator = tf.data.Dataset.from_generator(lambda:
                                                    map(tuple,
                                                        train_generator([val_path], [mask_val_path], test_img_augments,
                                                                        test_mask_augments, additional_path=path_lidar, additional_imgs=additional_imgs, output2=True if model_name == "RVN" else False)),
                                                    output_signature=output_signature
                                                    )

data_generator = data_generator.batch(BATCH_SIZE)
data_generator = data_generator.prefetch(buffer_size=tf.data.AUTOTUNE)
val_data_generator = val_data_generator.batch(BATCH_SIZE)
val_data_generator = val_data_generator.prefetch(buffer_size=tf.data.AUTOTUNE)

model.compile(optimizer=optimizer1, loss=_loss, metrics=_metrics)
    
if training:
    model.fit(data_generator,
              epochs=100,
              initial_epoch=train_num,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_steps=VALIDATION_STEPS,
              validation_data=val_data_generator,
              callbacks=callbacks,
              batch_size=BATCH_SIZE)

tf.keras.backend.clear_session()
