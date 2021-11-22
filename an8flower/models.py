import numpy as np

from keras import layers as klayers
from keras import models as kmodels
from kerasadf import layers as adflayers
from tensorflow.keras import layers, models


# GLOBAL DEFAULT PARAMETERS
MODELPATH = "an8flower-convnet-avgpool-weights.hdf5"
INPUT_SHAPE = (128, 128, 3)  # channels last, RGB 128 x 128 image


# STANDARD TF-KERAS MODELS
def load_model(path=MODELPATH, softmax=False):
    # input layer
    inputs = layers.Input(INPUT_SHAPE)
    # first convolution and pooling block
    conv1 = layers.Conv2D(
        32,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    pool1 = layers.AveragePooling2D(2)(conv1)
    # second convolution and pooling block
    conv2 = layers.Conv2D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    pool2 = layers.AveragePooling2D(2)(conv2)
    # third convolution and pooling block
    conv3 = layers.Conv2D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)
    pool3 = layers.AveragePooling2D(2)(conv3)
    # dense and output block
    flat = layers.Flatten()(pool3)
    dense1 = layers.Dense(
        1024, activation="relu", kernel_initializer="glorot_normal"
    )(flat)
    if softmax:
        activation = "softmax"
    else:
        activation = None
    dense2 = layers.Dense(
        12, activation=activation, kernel_initializer="glorot_normal"
    )(dense1)
    # build model
    model = models.Model(inputs, dense2)
    if MODELPATH is not None:
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
    conv1 = adflayers.Conv2D(
        32,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )([input_mean, input_var])
    pool1 = adflayers.AveragePooling2D(2, mode=mode)(conv1)
    # second convolution and pooling block
    conv2 = adflayers.Conv2D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(pool1)
    pool2 = adflayers.AveragePooling2D(2, mode=mode)(conv2)
    # third convolution and pooling block
    conv3 = adflayers.Conv2D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        mode=mode,
    )(pool2)
    pool3 = adflayers.AveragePooling2D(2, mode=mode)(conv3)
    # dense and output block
    flat = adflayers.Flatten(mode=mode)(pool3)
    dense1 = adflayers.Dense(
        1024, activation="relu", kernel_initializer="glorot_normal", mode=mode
    )(flat)
    dense2 = adflayers.Dense(
        12, kernel_initializer="glorot_normal", mode=mode
    )(dense1)
    # build model
    model = models.Model([input_mean, input_var], dense2)
    if MODELPATH is not None:
        model.load_weights(MODELPATH)
    return model


# STANDARD KERAS MODELS
def load_kmodel(path=MODELPATH, softmax=False):
    # input layer
    inputs = klayers.Input(INPUT_SHAPE)
    # first convolution and pooling block
    conv1 = klayers.Conv2D(
        32,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    pool1 = klayers.AveragePooling2D(2)(conv1)
    # second convolution and pooling block
    conv2 = klayers.Conv2D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    pool2 = klayers.AveragePooling2D(2)(conv2)
    # third convolution and pooling block
    conv3 = klayers.Conv2D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)
    pool3 = klayers.AveragePooling2D(2)(conv3)
    # dense and output block
    flat = klayers.Flatten()(pool3)
    dense1 = klayers.Dense(
        1024, activation="relu", kernel_initializer="glorot_normal"
    )(flat)
    if softmax:
        activation = "softmax"
    else:
        activation = None
    dense2 = klayers.Dense(
        12, activation=activation, kernel_initializer="glorot_normal"
    )(dense1)
    # build model
    model = kmodels.Model(inputs, dense2)
    if MODELPATH is not None:
        model.load_weights(MODELPATH)
    return model
