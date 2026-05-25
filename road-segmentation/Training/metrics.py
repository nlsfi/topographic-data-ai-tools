from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow import argmax


class F1Metric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    
    def __init__(self, **kwargs):
        super(F1Metric, self).__init__(name='f1metric')  # handles base args (e.g., dtype)
        self.tp = 1.
        self.fn = 1.
        self.fp = 1.
    
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp = K.sum(y_true * y_pred, axis=[1, 2])
        self.fn = K.sum(y_true * (1. - y_pred), axis=[1, 2])
        self.fp = K.sum((1. - y_true) * y_pred, axis=[1, 2])
        return (self.tp + K.epsilon()) / (self.tp + .5 * (self.fp + self.fn) + K.epsilon())
    
    def f1_score(self):
        tp = self.tp
        fn = self.fn
        fp = self.fp
        return (tp + K.epsilon()) / (tp + .5 * (fp + fn) + K.epsilon())
    
    def result(self):
        return self.f1_score()


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


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    tp = K.sum(y_true * y_pred, axis=[1, 2])
    fp = K.sum((1 - y_true) * y_pred, axis=[1, 2])
    fn = K.sum(y_true * (1 - y_pred), axis=[1, 2])
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    f1 = tf.clip_by_value(f1, 0., 1.)
    
    return K.mean(f1)


def iou_coef2(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[-1])
    union = K.sum(y_true, axis=[-1]) + K.sum(y_pred, axis=[-1]) - intersection
    iou = ((intersection + K.epsilon()) / (union + K.epsilon()))
    return iou


def f1_score2(y_true, y_pred):
    # y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
    # y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    tp = K.sum(y_true * y_pred, axis=[-1])
    fn = K.sum(y_true * (1. - y_pred), axis=[-1])
    fp = K.sum((1. - y_true) * y_pred, axis=[-1])
    return (tp + K.epsilon()) / (tp + .5 * (fp + fn) + K.epsilon())


def f1_score3(y_true, y_pred):
    # Exclude the last class (background) only when calculating F1 score
    y_true_f1 = y_true[:, :, :, :-1]
    y_pred_f1 = y_pred[:, :, :, :-1]
    
    tp = K.sum(y_true_f1 * y_pred_f1, axis=[-1])
    fn = K.sum(y_true_f1 * (1. - y_pred_f1), axis=[-1])
    fp = K.sum((1. - y_true_f1) * y_pred_f1, axis=[-1])
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return tf.reduce_mean(f1)


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


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())
