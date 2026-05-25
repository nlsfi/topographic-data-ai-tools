import tensorflow as tf
from keras.layers import *
from keras.models import Model
from tensorflow.keras.applications import VGG19

layers = tf.keras.layers


def bilinear_sampler(inputs, coords):
    """
    Bilinear sampling from inputs at coords.
    Args:
      inputs: Tensor (batch, H, W, C) to sample from.
      coords: Sampling grid (batch, H, W, 2) with normalized coords in [-1, 1].
              Last dimension represents (x, y) normalized coordinates.

    Returns:
      Sampled tensor of shape (batch, H, W, C)
    """
    batch_size, H, W, channels = tf.unstack(tf.shape(inputs))
    
    # Normalize coordinates to [0, H-1] and [0, W-1]
    x = coords[..., 0]
    y = coords[..., 1]
    x = (x + 1.0) * 0.5 * tf.cast(W - 1, tf.float32)
    y = (y + 1.0) * 0.5 * tf.cast(H - 1, tf.float32)
    
    x0 = tf.floor(x)
    x1 = x0 + 1
    y0 = tf.floor(y)
    y1 = y0 + 1
    
    # Clip to valid coordinates
    x0_safe = tf.clip_by_value(x0, 0, tf.cast(W - 1, tf.float32))
    x1_safe = tf.clip_by_value(x1, 0, tf.cast(W - 1, tf.float32))
    y0_safe = tf.clip_by_value(y0, 0, tf.cast(H - 1, tf.float32))
    y1_safe = tf.clip_by_value(y1, 0, tf.cast(H - 1, tf.float32))
    
    # Cast to int for gather
    x0_int = tf.cast(x0_safe, tf.int32)
    x1_int = tf.cast(x1_safe, tf.int32)
    y0_int = tf.cast(y0_safe, tf.int32)
    y1_int = tf.cast(y1_safe, tf.int32)
    
    # Calculate interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    # Gather pixel values at corner coords
    def gather_nd(params, x, y):
        batch_idx = tf.range(batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, H, W))
        
        indices = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(params, indices)
    
    Ia = gather_nd(inputs, x0_int, y0_int)
    Ib = gather_nd(inputs, x0_int, y1_int)
    Ic = gather_nd(inputs, x1_int, y0_int)
    Id = gather_nd(inputs, x1_int, y1_int)
    
    # Expand weights to channel dimension
    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)
    
    # Compute output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


def vgg_19(input):
    x_1 = conv_block(input, 64)
    x_1 = conv_block(x_1, 64)
    xp_1 = MaxPooling2D(pool_size=(2, 2))(x_1)
    x_2 = conv_block(xp_1, 128)
    x_2 = conv_block(x_2, 128)

    xp_2 = MaxPooling2D(pool_size=(2, 2))(x_2)
    x_3 = conv_block(xp_2, 256)
    x_3 = conv_block(x_3, 256)

    xp_3 = MaxPooling2D(pool_size=(2, 2))(x_3)
    x_4 = conv_block(xp_3, 512)
    x_4 = conv_block(x_4, 512)

    xp_4 = MaxPooling2D(pool_size=(2, 2))(x_4)
    x_5 = conv_block(xp_4, 512)

    return x_5, [x_1, x_2, x_3, x_4]


class BBDModule(tf.keras.layers.Layer):
    def __init__(self):
        super(BBDModule, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same')

    def call(self, F_do, X):
        X_cat = tf.concat([F_do, X], axis=-1)  # (batch, H, W, 2C)
        delta = self.conv(X_cat)  # (batch, H, W, 2)

        batch_size = tf.shape(X)[0]
        H = tf.shape(X)[1]
        W = tf.shape(X)[2]
        Hf = tf.cast(H, tf.float32)
        Wf = tf.cast(W, tf.float32)

        h = tf.linspace(-1.0, 1.0, H)
        w = tf.linspace(-1.0, 1.0, W)
        h_grid, w_grid = tf.meshgrid(h, w, indexing='ij')
        grid = tf.stack([w_grid, h_grid], axis=-1)  # (H, W, 2)
        grid = tf.expand_dims(grid, 0)  # (1, H, W, 2)
        norm = tf.stack([Wf, Hf])  # shape (2,)
        norm = tf.reshape(norm, [1, 1, 1, 2])  # shape (1, 1, 1, 2)
        delta_norm = delta / norm
        sampling_grid = grid + delta_norm  # (batch, H, W, 2)

        F_body = bilinear_sampler(X, sampling_grid)
        F_boundary = X - F_body
        return F_boundary, F_body


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


def sep_conv_bn_relu(x, filters):
    x = SeparableConv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(.2)(x)
    return x


def conv_bn_relu(x, filters):
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(.2)(x)
    return x


def conv_block(inputs, filters, dropout=False, conc=True):
    x = inputs
    if dropout:
        x = Dropout(0.25)(x)
    x = sep_conv_bn_relu(x, filters)
    x = sep_conv_bn_relu(x, filters)

    if conc:
        x_2 = conv_bn_relu(inputs, filters)
        x_2 = conv_bn_relu(x_2, filters)
        x = concatenate([x, x_2])
    x = squeeze_excite_block(x)
    return x


def encoder_imagenet(inputs, inputs2=None, second=False):
    skip_connections = []
    if inputs2 is not None:
        model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs2, classes=2)
    else:
        model = VGG19(include_top=False, weights=None, input_tensor=inputs, classes=2)
    f = True
    if second:
        model2 = tf.keras.Sequential()
        for layer in model.layers:
            layer._name += "second"
            model2.add(layer)
            if f:
                f = False
                layer._name = "input_1"
        
        names = ["block1_conv2second", "block2_conv2second", "block3_conv4second", "block4_conv4second"]
        output = model2.get_layer("block5_conv4second").output
        model2.summary()
    else:
        for layer in model.layers:
            layer._name += "first"
            if f:
                f = False
                layer._name = "input_1"
        
        names = ["block1_conv2first", "block2_conv2first", "block3_conv4first", "block4_conv4first"]
        output = model.get_layer("block5_conv4first").output
    for name in names:
        skip_connections.append(model.get_layer(name).output)
    
    model.trainable = True
    return output, skip_connections


def decoder(inputs, skip_connections):
    num_filters = [512, 256, 128, 64]
    
    skip_connections = skip_connections[::-1]
    skip_outputs = []
    x = inputs
    for i, f in enumerate(num_filters):
        x_in = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x_in, skip_connections[i]])
        x = conv_block(x, f)
        if i == 1:
            x_ddspp = DDSPP(x, f)
            x = Concatenate()([x, x_ddspp])
        x = conv_block(x, f)
        scale = 2 ** (len(num_filters) - i - 1)
        x_supervised = UpSampling2D((scale, scale), interpolation='bilinear')(x)
        skip_outputs.append(x_supervised)
    x = conv_block(x, f)
    
    return x, skip_outputs


def output_block(inputs, name: str ="output_1", sigmoid: bool = False):
    if sigmoid:
        x = Conv2D(2, (1, 1), padding="same", activation="sigmoid", name=name)(inputs)
    else:
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
    y1 = SeparableConv2D(filter, 1, padding="same")(y1)
    y1 = LeakyReLU(.2)(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)
    
    o1 = Concatenate()([x, y1])
    y2 = SeparableConv2D(filter, 3, dilation_rate=2, padding="same", use_bias=False)(o1)
    y2 = BatchNormalization()(y2)
    y2 = LeakyReLU(.2)(y2)
    
    o2 = Concatenate()([o1, y2])
    y3 = SeparableConv2D(filter, 3, dilation_rate=3, padding="same", use_bias=False)(o2)
    y3 = BatchNormalization()(y3)
    y3 = LeakyReLU(.2)(y3)
    
    o3 = Concatenate()([o2, y3])
    y4 = SeparableConv2D(filter, 3, dilation_rate=5, padding="same", use_bias=False)(o3)
    y4 = BatchNormalization()(y4)
    y4 = LeakyReLU(.2)(y4)
    
    o4 = Concatenate()([o3, y4])
    y5 = SeparableConv2D(filter, 3, dilation_rate=7, padding="same", use_bias=False)(o4)
    y5 = BatchNormalization()(y5)
    y5 = LeakyReLU(.2)(y5)
    
    o5 = Concatenate()([o4, y5])
    y6 = SeparableConv2D(filter, 3, dilation_rate=11, padding="same", use_bias=False)(o5)
    y6 = BatchNormalization()(y6)
    y6 = LeakyReLU(.2)(y6)
    
    y = Concatenate()([o5, y6])
    
    y = SeparableConv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(.2)(y)
    
    return y


def build_model(shape, deep_supervision=True):
    inputs = Input(shape)
    x, skip_1 = vgg_19(inputs)
    x = DDSPP(x, 64)
    x, decoder_skips = decoder(x, skip_1)
    if deep_supervision:
        outputs_ds1 = output_block(decoder_skips[0], "output_1_1", sigmoid=True)
        outputs_ds2 = output_block(decoder_skips[1], "output_1_2", sigmoid=True)
        outputs_ds3 = output_block(decoder_skips[2], "output_1_3", sigmoid=True)
        outputs_ds4 = output_block(decoder_skips[3], "output_1_4", sigmoid=True)
        conv_fuse = concatenate([*decoder_skips])
        outputs_ds5 = output_block(conv_fuse, "output_1_5", sigmoid=False)

    else:
        outputs1 = output_block(x, name='output_1_1')
        
    x = Concatenate()([inputs, x])
    
    x, skip_2 = encoder2(x)
    x = DDSPP(x, 64)
    x = decoder2(x, skip_2)
    outputs2 = output_block(x, name='output_2_1')
    if deep_supervision:
        model = Model(inputs, [outputs_ds1, outputs_ds2, outputs_ds3, outputs_ds4, outputs_ds5, outputs2])
    else:
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
