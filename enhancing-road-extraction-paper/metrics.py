from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow import argmax


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2]) - intersection
    iou = ((intersection + K.epsilon()) / (union + K.epsilon()))
    return iou


def f1_score(y_true, y_pred):
    tp = K.sum(y_true * y_pred, axis=[1, 2])
    fn = K.sum(y_true * (1. - y_pred), axis=[1, 2])
    fp = K.sum((1. - y_true) * y_pred, axis=[1, 2])
    return (tp + K.epsilon()) / (tp + .5 * (fp + fn) + K.epsilon())


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras

