import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_dilation, binary_erosion


def preprocess_additional_input(img):
    # set values below 0 to -1 and set the values between 0 and 1
    img = np.array(img)
    np.where(img < 0, -1, img)
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
