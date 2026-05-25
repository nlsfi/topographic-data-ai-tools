import os
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scipy.ndimage import binary_dilation, binary_erosion
from typing import List
import random


def preprocess_additional_input(img):
    # set values below 0 to -1 and set the values between 0 and 1
    img = np.array(img)
    if len(img.shape) == 2:
        img = img.reshape((*img.shape, 1))
    img = np.where(img < 0, -1, img)
    if (img.max() - img.min()) != 0:
        img = (img - img.min()) / (img.max() - img.min())
    return img


def preprocess_mask_image(image, color_limit):
    pic = np.array(image)
    img = np.zeros((pic.shape[0], pic.shape[1], 1))
    np.place(img[:, :, 0], pic[:, :, 0] >= color_limit, 1)
    return img


def generate_edges(_arr, output_img_size):
    """
    Generates edge mask from surface mask
    :param _arr: surface mask
    :param output_img_size: size of the output image, [x, y, c]
    :return: edge mask
    """
    gradients = [np.zeros((output_img_size[0], output_img_size[1], 1))]
    for z in range(1, _arr.shape[-1]):
        edge = np.array(_arr[:, :, z]).reshape([output_img_size[0], output_img_size[1]])
        edge = edge.astype('uint8')
        dilation = binary_dilation(edge, iterations=1)
        erosion = binary_erosion(dilation, iterations=1)
        gradients.append(np.array(np.bitwise_xor(dilation, erosion).reshape(output_img_size[0], output_img_size[1], 1)))
    gradients[0] = np.where(gradients[1] > 0, 1, 0)
    g = tf.one_hot(gradients[0], 2)
    g = np.array(g).reshape(output_img_size)
    return g


def data_generator(img_dirs, mask_dirs, dtm_path: str | None, augm, img_augmentations:List, mask_augmentations: List,
                   img_size=(512,512, 3), num_classes: int = 2, nir=False):
    list_images = os.listdir(img_dirs[0])
    if dtm_path:
        list_dtm = os.listdir(dtm_path)
        im_files = [im.split('.')[0] for im in list_images]
        im_end = list_images[0].split('.')[1]
        dtm_files = [im.split('.')[0] for im in list_dtm]
        dtm_end = list_dtm[0].split('.')[1]
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

        arr = np.array(arr)
        if arr.shape[-1] == 4 and not nir:
            arr = arr[:, :, :-1]
        if dtm_path:
            dtm = np.load(os.path.join(dtm_path, list_images[cur_ind].split('.')[0] + '.' + dtm_end))
            dtm = dtm[dtm.files[0]]
            dtm = preprocess_additional_input(dtm)
            arr = np.concatenate([arr, dtm], axis=-1)
        
        # add augments
        while arr.shape[-1] > img_size[-1]:
            arr = arr[:, :, :-1]
        for aug in img_augmentations:
            # arr = tf.image.resize(arr, (1000, 1000))
            arr = aug(arr)
        arr = np.array(arr)
        arr[:, :, :-1] = augm.normalize_data(arr[:, :, :-1])
        
        mask = np.array(Image.open(mask_dirs[0] + list_images[cur_ind].split(".")[0] + '.png'))
        
        # check ig mask shape has additional information
        mask_shape = mask.shape
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask_arr = mask.reshape((mask_shape[0], mask_shape[1], 1))
        
        # add mask augmentations
        for aug in mask_augmentations:
            mask_arr = aug(mask_arr)
        mask_arr = np.array(mask_arr)
        mask_arr = np.where(mask_arr > 80, 1, 0)
        
        mask_arr = tf.cast(mask_arr, tf.int32)
        mask_arr = tf.one_hot(mask_arr, 2)
        mask_arr = tf.reshape(mask_arr, (img_size[0], img_size[1], num_classes))
        
        augm.update_randoms()
        
        grads = generate_edges(mask_arr, (img_size[0], img_size[1], num_classes))
        
        img_i += 1
        yield (np.array(arr)), (
            np.array(mask_arr), np.array(mask_arr), np.array(mask_arr), np.array(mask_arr), np.array(mask_arr),
            np.array(grads))  # x_batches, y_batches  # output_dir, output_dir_masks


def get_data_generator(image_path: List[str], mask_path: List[str], dtm_path: str | None,
                       augm, img_augments, mask_augments,
                       img_size: tuple[int, int, int] = (512,512,3), num_classes: int = 2, use_nir: bool = False):
    
    output_signature = (
        (tf.TensorSpec(shape=img_size, dtype=tf.float32)),
        (tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32),
         tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32),
         tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32),
         tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32),
         tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32),
         tf.TensorSpec(shape=(512, 512, 2), dtype=tf.float32))
    )
    
    gen = tf.data.Dataset.from_generator(lambda:
                                   map(tuple, data_generator(image_path, mask_path, dtm_path,
                                                              augm, img_augments, mask_augments,
                                                              img_size, num_classes,
                                                              nir=use_nir)),
                                   output_signature=output_signature
                                   )
    return gen
