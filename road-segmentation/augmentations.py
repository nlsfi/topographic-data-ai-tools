import random
from scipy.ndimage import rotate
import tensorflow as tf
from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageFilter import (
    EDGE_ENHANCE, SHARPEN
)
from PIL import ImageFilter, Image
from skimage.exposure import match_histograms
import cv2
import os
import math
from scipy.ndimage import rotate


class Augmentation:
    def __init__(self, img_size, crop_size=(512,512), seed=(1234, 2345), max_delta_brightness=.25, rand_contrast_values=(.6, 1.),
                 resize_method='bilinear'):
        
        self.rand_flip_lr = None
        self.rand_flip_ud = None
        self.rand1 = None
        self.rand2 = None
        self.rg = None
        self.resize_method = resize_method
        self.seed = seed
        self.max_delta_brightness = max_delta_brightness
        self.rand_contrast_lower = rand_contrast_values[0]
        self.rand_contrast_upper = rand_contrast_values[1]
        self.img_size = img_size
        self.update_randoms()
    
    def update_randoms(self, arr=None):
        self.rand1 = np.random.randint(0, 1000, dtype=int)
        self.rand2 = np.random.randint(0, 1000, dtype=int)
        self.rand_flip_lr = np.random.random()
        self.rand_flip_ud = np.random.random()
        self.seed = (self.rand1, self.rand2)
        self.rg = random.randint(0, 360)
        return arr
    
    def random_crop(self, arr, size=None):
        if size:
            arr = tf.image.stateless_random_crop(arr, size, seed=self.seed)
        else:
            arr = tf.image.stateless_random_crop(arr, (self.img_size[0], self.img_size[1], arr.shape[-1]),
                                                 seed=self.seed)
        return arr
    
    def crop(self, arr, crop_values=(0, 0, 512, 512)):
        arr = arr[:512, :512, :]
        return arr
    
    def random_crop_1000x1000(self, arr):
        arr = tf.image.stateless_random_crop(arr, (1000, 1000, tf.shape(arr)[-1]), seed=self.seed)
        return arr
    
    def random_crop_1024x1024(self, arr):
        arr = tf.image.stateless_random_crop(arr, (1024, 1024, tf.shape(arr)[-1]), seed=self.seed)
        return arr
    
    def crop_1000x1000(self, arr, crop_values=(0, 0, 1000, 1000)):
        arr = tf.image.crop_to_bounding_box(arr, crop_values[0], crop_values[1], crop_values[2], crop_values[3])
        return arr
    
    def resize(self, arr, size=None, method=None):
        """if size:
            arr = tf.image.resize(arr, [size[0], size[1]], method=self.resize_method)
        else:"""
        #arr = np.resize(arr, (512, 512, arr.shape[-1]))
        if not method:
            arr = tf.image.resize(arr, [512, 512], method=self.resize_method)
        else:
            arr = tf.image.resize(arr, [512, 512], method=method)
        return arr
    
    def mask_resize(self, arr, size=None):
        if size:
            arr = tf.image.resize(arr, [size[0], size[1]], method='nearest')
        else:
            arr = tf.image.resize(arr, [self.img_size[0], self.img_size[1]], method='nearest')
        return arr
    
    def random_flip_up_down(self, arr):
        if self.rand_flip_ud > .5:
            arr = tf.image.flip_up_down(arr)
        return arr
    
    def random_flip_left_right(self, arr):
        if self.rand_flip_lr > .5:
            arr = tf.image.flip_left_right(arr)
        return arr
    
    def normalize_data(self, arr):
        arr = np.array(arr)
        minimum = np.min(arr)
        maximum = np.max(arr)
        if maximum > minimum:
            output = np.divide(np.subtract(arr, minimum), np.subtract(maximum, minimum))
            return output
        else:
            return arr
    
    def edge_enhance(self, arr):
        arr = np.array(arr, dtype=np.uint8)
        arr = Image.fromarray(arr)
        arr = arr.filter(EDGE_ENHANCE)
        arr = np.array(arr)
        return arr
    
    def sharpen(self, arr):
        arr = np.array(arr, dtype=np.uint8)
        arr = Image.fromarray(arr)
        
        arr = arr.filter(SHARPEN)
        arr = np.array(arr)
        return arr
    
    def random_brightness(self, arr):
        arr = tf.image.random_brightness(
            arr, self.max_delta_brightness, seed=None
        )
        return arr
    
    def random_contrast(self, arr):
        arr = tf.image.random_contrast(
            arr, self.rand_contrast_lower, self.rand_contrast_upper
        )
        return arr
    
    """def random_rotate(self, arr):
        #arr = np.array(ndimage.rotate(arr, self.rg,
        #               mode='grid-mirror'))
        arr = tf.keras.layers.RandomRotation(.3, 'reflect', 'bilinear', self.rg)(arr)
        arr = np.array(arr, dtype=float)
        return arr"""

    """def random_rotate(self, img):
        print('RR')
        print(img.shape)
        # Ensure the input is a numpy array
        img = np.array(img)
    
        # Generate a random angle between 0 and 359 degrees
        angle = self.rg  # Assuming self.rg is defined elsewhere
    
        # Get the image dimensions
        (h, w) = img.shape[:2]
    
        # Calculate the center of the image
        center = (w // 2, h // 2)
    
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
        # Perform the rotation
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))
    
        print(rotated_img.shape)
        if len(rotated_img.shape) == 2:
            rotated_img = np.reshape(rotated_img, (*rotated_img.shape, 1))
        return rotated_img"""

    def random_rotate(self, img):
        # Ensure the input is a numpy array
        img = np.array(img)
    
        # Generate a random angle between 0 and 359 degrees
        angle = self.rg
    
        # Rotate the image using scipy.ndimage.rotate
        rotated_img = rotate(img, angle, reshape=False, mode='grid-mirror')
    
        return rotated_img
    
    def color_histogram_matching(self, source, template):
        return match_histograms(source, template, channel_axis=-1)
    
    def adaptive_color_histogram_matching(self, source, template):
        source_region = cv2.cvtColor(source, cv2.COLOR_RGB2YCrCb)
        ref_region = cv2.cvtColor(template, cv2.COLOR_RGB2YCrCb)
        source_region[:, :, 0] = cv2.equalizeHist(source_region[:, :, 0].astype(np.uint8))
        ref_region[:, :, 0] = cv2.equalizeHist(ref_region[:, :, 0].astype(np.uint8))
        source_region = cv2.cvtColor(source_region, cv2.COLOR_YCrCb2RGB)
        ref_region = cv2.cvtColor(ref_region, cv2.COLOR_YCrCb2RGB)
        source_region = source_region.astype(float)
        ref_region = ref_region.astype(float)
        source_region = np.clip(source_region, 0, 1)
        ref_region = np.clip(ref_region, 0, 1)
        return source_region, ref_region
    
    
    def rgb_to_hsv(self, arr):
        arr = cv2.cvtColor(np.array(arr), cv2.COLOR_RGB2HSV)
        return arr
    
    def rgb_to_ycbcr(self, arr):
        arr = cv2.cvtColor(np.array(arr), cv2.COLOR_RGB2YCrCb)
        return arr
    
    def rgb_to_yuv(self, arr):
        arr = cv2.cvtColor(np.array(arr), cv2.COLOR_RGB2YUV)
        return arr
    
    def rgb_to_lab(self, arr):
        arr = cv2.cvtColor(np.array(arr), cv2.COLOR_RGB2LAB)
        return arr
    
    def rgb_to_xyz(self, arr):
        arr = cv2.cvtColor(np.array(arr), cv2.COLOR_RGB2XYZ)
        return arr
    
    def rgb_to_cmy(self, arr):
        arr = np.divide(arr, 255.)
        arr = np.subtract(1., arr)
        return arr
