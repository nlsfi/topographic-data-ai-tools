"""
https://github.com/Arnick8/RoadVecNet
"""
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from tensorflow.keras.applications import VGG19
import numpy as np
layers = tf.keras.layers

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    x = Multiply()([init, se])
    return x


def conv_block(inputs, filters):
    x = inputs
    
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = squeeze_excite_block(x)
    
    return x


def encoder1(inputs, inputs2=None, second=False):
    skip_connections = []
    if inputs2 is not None:
        model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs2, classes=2)
    else:
        model = VGG19(include_top=False, weights=None, input_tensor=inputs, classes=2)
    f = True
    if second:
        model2 = tf.keras.Sequential()
        for layer in model.layers:
            layer.name += "second"
            model2.add(layer)
            if f:
                f = False
                layer.name = "input_1"
        
        names = ["block1_conv2second", "block2_conv2second", "block3_conv4second", "block4_conv4second"]
        output = model2.get_layer("block5_conv4second").output
        # output = Model(model.input, model.layers[-7].output)
        model2.summary()
    else:
        for layer in model.layers:
            layer.name += "first"
            if f:
                f = False
                layer.name = "input_1"
        
        names = ["block1_conv2first", "block2_conv2first", "block3_conv4first", "block4_conv4first"]
        output = model.get_layer("block5_conv4first").output
    for name in names:
        skip_connections.append(model.get_layer(name).output)
    
    # output = model.get_layer("block5_conv4").output
    model.trainable = True
    return output, skip_connections


def decoder(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    
    skip_connections.reverse()
    x = inputs
    
    for i, f in enumerate(num_filters):
        
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        
        x = Conv2D(f, 3, 1, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = squeeze_excite_block(x)
        
        x = Conv2D(num_filters[i], 3, 1, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        x = Conv2D(num_filters[i], 3, 1, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    
    return x


def output_block(inputs, name="output_1"):
    x = Conv2D(2, (1, 1), padding="same", activation="softmax", name=name)(inputs)
    return x


def output_block1(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    return x

def output_block3(inputs, name="output_1"):
    x = Conv2D(3, (1, 1), padding="same", activation="softmax", name=name)(inputs)
    return x


def Upsample(tensor, size):
    """Bilinear upsampling"""
    
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)


def DDSPP(x, filter):
    shape = x.shape
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)
    
    y2 = Conv2D(filter, 3, dilation_rate=2, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    
    y3 = Conv2D(filter, 3, dilation_rate=4, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    
    y4 = Conv2D(filter, 3, dilation_rate=8, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = ReLU()(y4)
    
    y5 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = ReLU()(y5)
    
    y = Concatenate()([x, y1, y2, y3, y4, y5])
    
    y = Conv2D(filter, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    
    return y


def build_model2(shape):
    inputs = Input(shape)
    x, skip_1 = encoder1(inputs)
    x = DDSPP(x, 64)
    x = decoder(x, skip_1)
    outputs1 = output_block(x, name='output_1')
    x = Concatenate()([inputs, outputs1])
    
    x, skip_2 = encoder2(x)
    x = DDSPP(x, 64)
    x = decoder2(x, skip_2)
    outputs2 = output_block(x, name='output_2')
    model = Model(inputs, [outputs1, outputs2])
    
    return model


def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs
    
    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)
    
    return x, skip_connections


def decoder2(inputs, skip_2):
    num_filters = [256, 128, 64, 32]
    skip_2.reverse()
    x = inputs
    
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_2[i]])
        x = conv_block(x, f)
    
    return x
