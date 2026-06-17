import random
import numpy as np
import tensorflow as tf
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image
from skimage.exposure import match_histograms
from scipy.ndimage import rotate


class Augmentation:
    def __init__(self, img_size, crop_size=(512,512), seed=(1234, 2345), resize_method='bilinear'):
        self.rand_flip_lr = None
        self.rand_flip_ud = None
        self.rand1 = None
        self.rand2 = None
        self.rg = None
        self.crop_size = crop_size
        self.resize_method = resize_method
        self.seed = seed
        self.img_size = img_size
        self.update_randoms()
    
    def update_randoms(self):
        self.rand1 = np.random.randint(0, 1000, dtype=int)
        self.rand2 = np.random.randint(0, 1000, dtype=int)
        self.rand_flip_lr = np.random.random()
        self.rand_flip_ud = np.random.random()
        self.seed = (self.rand1, self.rand2)
        self.rg = random.randint(0, 360)
        
    def normalize_data(self, arr):
        return arr / 255.
    
    def random_crop(self, arr, size=None):
        if size:
            arr = tf.image.stateless_random_crop(arr, size, seed=self.seed)
        else:
            arr = tf.image.stateless_random_crop(arr, (self.img_size[0], self.img_size[1], arr.shape[-1]),
                                                 seed=self.seed)
        return arr
    
    def crop(self, arr):
        arr = arr[:self.crop_size[0], :self.crop_size[1], :]
        return arr
    
    def resize(self, arr):
        arr = tf.image.resize(arr, [512, 512], method=self.resize_method)
        return arr
    
    def random_flip_up_down(self, arr):
        if self.rand_flip_ud > .5:
            arr = tf.image.flip_up_down(arr)
        return arr
    
    def random_flip_left_right(self, arr):
        if self.rand_flip_lr > .5:
            arr = tf.image.flip_left_right(arr)
        return arr
    
    def random_rotate(self, img):
        img = np.array(img)
        angle = self.rg
        rotated_img = rotate(img, angle, reshape=False, mode='grid-mirror')
        return rotated_img
