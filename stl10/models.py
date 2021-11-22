import numpy as np

from keras import layers as klayers
from keras import models as kmodels
from kerasadf import layers as adflayers
from tensorflow.keras import layers, models


# GLOBAL DEFAULT PARAMETERS
MODELPATH = "stl10-vgg16-avgpool-weights.hdf5"
INPUT_SHAPE = (224, 224, 3)  # channels last, 3 colors 224 x 224


# STANDARD TF-KERAS MODELS
def load_model(path=MODELPATH, softmax=False):
    # input layer
    inputs = layers.Input(INPUT_SHAPE)
    # first convolution and pooling block
    conv1a = layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    conv1b = layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1a)
    pool1 = layers.AveragePooling2D(2)(conv1b)
    # second convolution and pooling block
    conv2a = layers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    conv2b = layers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv2a)
    pool2 = layers.AveragePooling2D(2)(conv2b)
    # third convolution and pooling block
    conv3a = layers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)
    conv3b = layers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3a)
    conv3c = layers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3b)
    pool3 = layers.AveragePooling2D(2)(conv3c)
    # fourth convolution and pooling block
    conv4a = layers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool3)
    conv4b = layers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4a)
    conv4c = layers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4b)
    pool4 = layers.AveragePooling2D(2)(conv4c)
    # fifth convolution and pooling block
    conv5a = layers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool4)
    conv5b = layers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv5a)
    conv5c = layers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv5b)
    pool5 = layers.AveragePooling2D(2)(conv5c)
    # dense and output block
    flat = layers.Flatten()(pool5)
    dense1 = layers.Dense(
        4096, activation="relu", kernel_initializer="glorot_normal"
    )(flat)
    dense2 = layers.Dense(
        4096, activation="relu", kernel_initializer="glorot_normal"
    )(dense1)
    if softmax:
        activation = "softmax"
    else:
        activation = None
    dense3 = layers.Dense(
        10, activation=activation, kernel_initializer="glorot_normal"
    )(dense2)
    # build model
    model = models.Model(inputs, dense3)
    model.load_weights(MODELPATH)
    return model


# ADF TF-KERAS MODELS
def load_adfmodel(path=MODELPATH, mode="diag", rank=None):
    # input layer
    input_mean = layers.Input(INPUT_SHAPE)
    if mode == "diag":
        input_var = layers.Input(INPUT_SHAPE)
    elif mode == "half":
        if rank is None:
            rank = np.prod(INPUT_SHAPE)
        input_var = layers.Input([rank] + list(INPUT_SHAPE))
    elif mode == "full":
        input_var = layers.Input(list(INPUT_SHAPE) * 2)
    # first convolution and pooling block
    conv1a = adflayers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )([input_mean, input_var])
    conv1b = adflayers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(conv1a)
    pool1 = adflayers.AveragePooling2D(2, mode=mode)(conv1b)
    # second convolution and pooling block
    conv2a = adflayers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(pool1)
    conv2b = adflayers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(conv2a)
    pool2 = adflayers.AveragePooling2D(2, mode=mode)(conv2b)
    # third convolution and pooling block
    conv3a = adflayers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(pool2)
    conv3b = adflayers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(conv3a)
    conv3c = adflayers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(conv3b)
    pool3 = adflayers.AveragePooling2D(2, mode=mode)(conv3c)
    # fourth convolution and pooling block
    conv4a = adflayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(pool3)
    conv4b = adflayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(conv4a)
    conv4c = adflayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(conv4b)
    pool4 = adflayers.AveragePooling2D(2, mode=mode)(conv4c)
    # fifth convolution and pooling block
    conv5a = adflayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(pool4)
    conv5b = adflayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(conv5a)
    conv5c = adflayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(conv5b)
    pool5 = adflayers.AveragePooling2D(2, mode=mode)(conv5c)
    # dense and output block
    flat = adflayers.Flatten(mode=mode)(pool5)
    dense1 = adflayers.Dense(
        4096, activation="relu", kernel_initializer="glorot_normal", mode=mode
    )(flat)
    dense2 = adflayers.Dense(
        4096, activation="relu", kernel_initializer="glorot_normal", mode=mode
    )(dense1)
    dense3 = adflayers.Dense(
        10, kernel_initializer="glorot_normal", mode=mode
    )(dense2)
    # build model
    model = models.Model([input_mean, input_var], dense3)
    model.load_weights(MODELPATH)
    return model


# STANDARD KERAS MODELS
def load_kmodel(path=MODELPATH, softmax=False):
    # input layer
    inputs = klayers.Input(INPUT_SHAPE)
    # first convolution and pooling block
    conv1a = klayers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    conv1b = klayers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1a)
    pool1 = klayers.AveragePooling2D(2)(conv1b)
    # second convolution and pooling block
    conv2a = klayers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    conv2b = klayers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv2a)
    pool2 = klayers.AveragePooling2D(2)(conv2b)
    # third convolution and pooling block
    conv3a = klayers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)
    conv3b = klayers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3a)
    conv3c = klayers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3b)
    pool3 = klayers.AveragePooling2D(2)(conv3c)
    # fourth convolution and pooling block
    conv4a = klayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool3)
    conv4b = klayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4a)
    conv4c = klayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4b)
    pool4 = klayers.AveragePooling2D(2)(conv4c)
    # fifth convolution and pooling block
    conv5a = klayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool4)
    conv5b = klayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv5a)
    conv5c = klayers.Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv5b)
    pool5 = klayers.AveragePooling2D(2)(conv5c)
    # dense and output block
    flat = klayers.Flatten()(pool5)
    dense1 = klayers.Dense(
        4096, activation="relu", kernel_initializer="glorot_normal"
    )(flat)
    dense2 = klayers.Dense(
        4096, activation="relu", kernel_initializer="glorot_normal"
    )(dense1)
    if softmax:
        activation = "softmax"
    else:
        activation = None
    dense3 = klayers.Dense(
        10, activation=activation, kernel_initializer="glorot_normal"
    )(dense2)
    # build model
    model = kmodels.Model(inputs, dense3)
    model.load_weights(MODELPATH)
    return model
