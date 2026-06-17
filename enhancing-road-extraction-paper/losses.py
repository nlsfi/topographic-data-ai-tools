import metrics
import numpy as np
import tensorflow as tf
from typing import Callable, Union
import tensorflow.keras.backend as K
from tensorflow import convert_to_tensor as _to_tensor
from scipy.ndimage import distance_transform_edt as distance


def dice_coef_loss(y_true, y_pred):
    return 1. - metrics.dice_coef(y_true, y_pred)


def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = _to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = K.log(prediction_tensor / (1 - prediction_tensor))
    
    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * tf.cast(
            tf.sigmoid(prediction_tensor) > 0.5, tf.float32)
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))


def dice_coef_loss_bce(y_true, y_pred, bootstrapping = "soft"):
    dice = .5
    bce = .5
    alpha = .95
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice
