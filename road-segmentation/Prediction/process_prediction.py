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
import argparse
from pathlib import Path
import cv2
from .combine_images import pad_image_with_neighbours, num_neighbours, get_neighbour_ids
import geopandas as gpd
from Training import preprocess_additional_input, preprocess_mask_image


def prepare_large_image(images, masks, augm, img_augs, img_size):
    x_batches = [[] for _ in range(len(images))]
    y_batches = [[] for _ in range(len(masks))]
    
    for i, arr1 in enumerate(images):
        for aug in img_augs:
            arr1 = aug(arr1)
        arr1 = np.array(arr1)
        arr1[:, :, :-1] = augm.normalize_data(arr1[:, :, :-1])
        arr = tf.reshape(arr1, img_size)
        x_batches[i] += [arr]
        
    for i in range(len(masks)):
        mask = np.array(masks[i].copy())
        mask_shape = mask.shape
        if len(mask.shape) == 3:
            mask = mask[:, :, 1]
        mask = mask.reshape((mask_shape[0], mask_shape[1], 1))
        mask_arr = preprocess_mask_image(mask, 2)
        
        mask_arr = tf.cast(mask_arr, tf.int32)
        mask_arr = tf.one_hot(mask_arr, 1)
        mask_arr = tf.reshape(mask_arr, (img_size[0], img_size[1], 1))
        y_batches[i] += [mask_arr]
        y_batches[i] = np.array(y_batches[i])
        
    input_dict = {"input_layer": np.stack(x_batch) for x_batch, _ in zip(x_batches, range(len(images)))}
    output_dict = {"output_{}".format(str(_ + 1)): y_batch for y_batch, _ in zip(y_batches, range(len(masks)))}
    return input_dict, output_dict


def predict_crop(model, x, y, input_img, img_size, cut_shape, im_mask, im_mask2, mask_shape, augm, img_augmentations,
                 cut, output1, step):
    input_img = np.array(input_img)
    
    im_test, im_masks = prepare_large_image([input_img[x:x + cut_shape, y:y + cut_shape, :]],
                                            [im_mask[0:mask_shape, 0:mask_shape],
                                             im_mask2[0:mask_shape, 0:mask_shape]],
                                            augm, img_augmentations, img_size)
    
    im_test['input_layer'] = tf.reshape(im_test['input_layer'], (1, mask_shape, mask_shape, img_size[-1]))
    
    out_test = model.predict(im_test["input_layer"], verbose=0)
    
    out1 = tf.reshape(out_test[-2], (mask_shape, mask_shape, 2))
    out1 = tf.image.resize(out1, [cut_shape, cut_shape], method='bilinear')
    out1 = np.array(out1)
    out1 = out1[:, :, 1]
    out1 = out1.reshape((out1.shape[0], out1.shape[1], 1))

    if cut == 0:
        output1[x:x + out1.shape[0] - cut, y + cut:y + out1.shape[1] - cut, :] = out1[:, :, :]
    else:
        if (step <= x < input_img.shape[1] - cut_shape) and \
                (step <= y < input_img.shape[0] - cut_shape):
            # in the mid-area of the large image, so we can cut from every direction
            output1[x + cut:x + out1.shape[0] - cut, y + cut:y + out1.shape[1] - cut, :] = out1[
                                                                                           cut:-cut,
                                                                                           cut:-cut, :]
        elif (step <= x < input_img.shape[1] - cut_shape) and not \
                (step <= y < input_img.shape[0] - cut_shape):
            if step <= y:
                output1[x + cut:x + out1.shape[0] - cut, y + cut:y + out1.shape[1], :] = out1[cut:-cut,
                                                                                         cut:, :]
            elif y < input_img.shape[0] - cut_shape:
                output1[x + cut:x + out1.shape[0] - cut, y:y + out1.shape[1] - cut, :] = out1[cut:-cut,
                                                                                         :-cut,
                                                                                         :]
            else:
                output1[x + cut:x + out1.shape[0] - cut, y:y + out1.shape[1], :] = out1[cut:-cut, :, :]

        elif not (step <= x < input_img.shape[1] - cut_shape) and \
                (step <= y < input_img.shape[0] - cut_shape):
            if step <= x:
                output1[x + cut:x + out1.shape[0], y + cut:y + out1.shape[1] - cut, :] = out1[cut:,
                                                                                         cut:-cut, :]
            elif x < input_img.shape[1] - cut_shape:
                output1[x:x + out1.shape[0] - cut, y + cut:y + out1.shape[1] - cut, :] = out1[:-cut,
                                                                                         cut:-cut,
                                                                                         :]
            else:
                output1[x:x + out1.shape[0], y + cut:y + out1.shape[1] - cut, :] = out1[:, cut:-cut, :]
        
        else:
            if x == 0:
                if y == 0:
                    output1[x:x + out1.shape[0], y:y + out1.shape[1], :] = out1[:, :, :]
                else:
                    output1[x:x + out1.shape[0], y + cut:y + out1.shape[1], :] = out1[:, cut:, :]
            else:
                if y == 0:
                    output1[x + cut:x + out1.shape[0], y:y + out1.shape[1], :] = out1[cut:, :, :]
                else:
                    output1[x:x + out1.shape[0], y + cut:y + out1.shape[1], :] = out1[:, cut:, :]

    return output1


def large_prediction(model, img_path, augm, img_augmentations, img_size, additional_channel_path=None, output_path=None):
    # Get images
    folders = os.listdir(additional_channel_path)
    gpkg_file = "TM35_karttalehtijako.gpkg"
    gpkg = gpd.read_file(gpkg_file, layer="utm5")
    for folder in folders[:]:
        images = [folder.split('_')[0]]
        folder_files = os.listdir(additional_channel_path)
        print(images)
        if not output_path:
            output_path = "outputs/"
        os.makedirs(output_path, exist_ok=True)
        # check if outputs already exist
        finished = os.listdir(output_path)
        finished = [f.split('.')[0] for f in finished]
        for img_name in images:
            if img_name.split(".")[0] in finished:
                continue
            if img_name.split(".")[-1].lower() not in {"tif", "tiff"}:
                continue
            print(f"Current img:, {img_name}")
            # check if we have neighbouring image areas
            neighbours = get_neighbour_ids(img_name, gpkg)
            num_n = num_neighbours(neighbours, folder_files)
            if num_n != 8:
                continue
            img_name = img_name.split(".")[0]
            img = pad_image_with_neighbours(img_path, img_name + ".tif", pad_size=1000, channels=[1, 2, 3], gpkg=gpkg)
            img = img.transpose(1, 2, 0)
            img = np.array(img).astype(int)
            
            if additional_channel_path is not None:
                dtm = pad_image_with_neighbours(additional_channel_path, img_name + ".tif", pad_size=1000, channels=1, gpkg=gpkg)
                dtm = preprocess_additional_input(dtm)
                img = np.concatenate([img, dtm], axis=-1)
            
            cut_shape = 512 * 2
            mask_shape = 512
            cut = 128 * 2
            effective_size = cut_shape - cut * 2
            step = effective_size  # Actual step between tile centers
            
            # Calculate the padding needed to make the image size divisible by step
            padding_x = (effective_size - (img.shape[0] % effective_size)) % effective_size
            padding_y = (effective_size - (img.shape[1] % effective_size)) % effective_size
            # Pad the image
            img = np.pad(img, ((padding_x // 2, padding_x // 2), (padding_y // 2, padding_y // 2), (0, 0)),
                              'symmetric')
            shape = img.shape
            
            im_mask = np.zeros((shape[0] + 48, shape[1] + 48))
            im_mask2 = np.zeros((shape[0] + 48, shape[1] + 48))
            
            output1 = np.zeros((shape[0], shape[1], 1))
            
            if img.shape[-1] > img_size[-1]:
                img = img[:, :, :-1]
            starting_time = time.time()
            for x in range(0, shape[0] - cut_shape + 1, step):
                round_time = time.time()
                for y in range(0, shape[1] - cut_shape + 1, step):
                    output1 = predict_crop(model, x, y, img, img_size, cut_shape, im_mask, im_mask2, mask_shape, augm,
                                                    img_augmentations,
                                                    cut, output1, step)
                
                print(x, round_time - starting_time)
            output1 = predict_crop(model, shape[0] - cut_shape, shape[1] - cut_shape, img, img_size, cut_shape, im_mask,
                                            im_mask2, mask_shape, augm, img_augmentations,
                                            cut, output1, step)
            output1 = output1[padding_x // 2:-padding_x // 2, padding_y // 2:-padding_y // 2, :]
            output1 = output1.reshape((output1.shape[0], output1.shape[1])) * 255
            output1 = output1.astype('uint8')

            try:
                cv2.imwrite(
                    os.path.join(output_path, f"{img_name}.png"),
                    output1)
                print(
                    f"Output1: saved to {output_path}/{img_name}.png")

            except Exception as e:
                print('imsave failed', f"{output_path}/{img_name}.png")
                print(e)
                continue
            
            print("Saving..", time.time() - starting_time, time.time() - round_time)
