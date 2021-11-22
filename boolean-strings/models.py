import numpy as np

from keras import initializers as kinitializers
from keras import layers as klayers
from keras import models as kmodels
from kerasadf import layers as adflayers
from tensorflow.keras import initializers, layers, models


# GLOBAL DEFAULT PARAMETERS
DIMENSION = 16
BLOCK_SIZE = 5


# STANDARD TF-KERAS MODELS
def create_simple_model(dimension=DIMENSION, block_size=BLOCK_SIZE):
    inp = layers.Input(shape=(dimension, 1))
    blockwise_sums = layers.Conv1D(
        1,
        block_size,
        1,
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Constant(-(block_size - 1)),
        activation="relu",
    )(inp)
    flat = layers.Flatten()(blockwise_sums)
    full_sum = layers.Dense(
        1,
        kernel_initializer=initializers.Constant(-1),
        bias_initializer=initializers.Ones(),
        activation="relu",
    )(flat)
    out = layers.Dense(
        1,
        kernel_initializer=initializers.Constant(-1),
        bias_initializer=initializers.Ones(),
        activation="linear",
    )(full_sum)
    return models.Model(inp, out)


def create_scaled_model(width, dimension=DIMENSION, block_size=BLOCK_SIZE):
    inp = layers.Input(shape=(dimension, 1))
    shift_inp_layer = layers.Conv1D(2, 1, 1, activation="relu")
    shifted_inp = shift_inp_layer(inp)
    shift_inp_layer.set_weights(
        [
            np.asarray([[[1, 1]]]),
            np.asarray([-0.5 + width / 2, -0.5 - width / 2]),
        ]
    )
    scale_inp_layer = layers.Conv1D(1, 1, 1, activation="linear")
    scaled_inp = scale_inp_layer(shifted_inp)
    scale_inp_layer.set_weights(
        [np.asarray([[[1 / width], [-1 / width]]]), np.asarray([0])]
    )
    blockwise_sums = layers.Conv1D(
        1,
        block_size,
        1,
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Constant(-(block_size - 1)),
        activation="relu",
    )(scaled_inp)
    flat = layers.Flatten()(blockwise_sums)
    full_sum = layers.Dense(
        1,
        kernel_initializer=initializers.Constant(-1),
        bias_initializer=initializers.Ones(),
        activation="relu",
    )(flat)
    out = layers.Dense(
        1,
        kernel_initializer=initializers.Constant(-1),
        bias_initializer=initializers.Ones(),
        activation="linear",
    )(full_sum)
    return models.Model(inp, out)


# ADF TF-KERAS MODELS
def create_simple_adfmodel(
    dimension=DIMENSION, block_size=BLOCK_SIZE, mode="diag", rank=None
):
    inp_mean = layers.Input(shape=(dimension, 1))
    if mode == "diag":
        inp_var = layers.Input(shape=(dimension, 1))
    elif mode == "half":
        if rank is None:
            rank = DIMENSION
        inp_var = layers.Input(shape=(rank, dimension, 1))
    elif mode == "full":
        inp_var = layers.Input(shape=(dimension, 1, dimension, 1))
    blockwise_sums = adflayers.Conv1D(
        1,
        block_size,
        1,
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Constant(-(block_size - 1)),
        activation="relu",
        mode=mode,
    )([inp_mean, inp_var])
    flat = adflayers.Flatten(mode=mode)(blockwise_sums)
    full_sum = adflayers.Dense(
        1,
        kernel_initializer=initializers.Constant(-1),
        bias_initializer=initializers.Ones(),
        activation="relu",
        mode=mode,
    )(flat)
    out = adflayers.Dense(
        1,
        kernel_initializer=initializers.Constant(-1),
        bias_initializer=initializers.Ones(),
        activation="linear",
        mode=mode,
    )(full_sum)
    return models.Model([inp_mean, inp_var], out)


def create_scaled_adfmodel(
    width, dimension=DIMENSION, block_size=BLOCK_SIZE, mode="diag", rank=None
):
    inp_mean = layers.Input(shape=(dimension, 1))
    if mode == "diag":
        inp_var = layers.Input(shape=(dimension, 1))
    elif mode == "half":
        if rank is None:
            rank = DIMENSION
        inp_var = layers.Input(shape=(rank, dimension, 1))
    elif mode == "full":
        inp_var = layers.Input(shape=(dimension, 1, dimension, 1))
    shift_inp_layer = adflayers.Conv1D(2, 1, 1, activation="relu", mode=mode)
    shifted_inp = shift_inp_layer([inp_mean, inp_var])
    shift_inp_layer.set_weights(
        [
            np.asarray([[[1, 1]]]),
            np.asarray([-0.5 + width / 2, -0.5 - width / 2]),
        ]
    )
    scale_inp_layer = adflayers.Conv1D(1, 1, 1, activation="linear", mode=mode)
    scaled_inp = scale_inp_layer(shifted_inp)
    scale_inp_layer.set_weights(
        [np.asarray([[[1 / width], [-1 / width]]]), np.asarray([0])]
    )
    blockwise_sums = adflayers.Conv1D(
        1,
        block_size,
        1,
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Constant(-(block_size - 1)),
        activation="relu",
        mode=mode,
    )(scaled_inp)
    flat = adflayers.Flatten(mode=mode)(blockwise_sums)
    full_sum = adflayers.Dense(
        1,
        kernel_initializer=initializers.Constant(-1),
        bias_initializer=initializers.Ones(),
        activation="relu",
        mode=mode,
    )(flat)
    out = adflayers.Dense(
        1,
        kernel_initializer=initializers.Constant(-1),
        bias_initializer=initializers.Ones(),
        activation="linear",
        mode=mode,
    )(full_sum)
    return models.Model([inp_mean, inp_var], out)


# STANDARD KERAS MODELS
def create_simple_kmodel(dimension=DIMENSION, block_size=BLOCK_SIZE):
    inp = klayers.Input(shape=(dimension, 1))
    blockwise_sums = klayers.Conv1D(
        1,
        block_size,
        strides=1,
        kernel_initializer=kinitializers.Ones(),
        bias_initializer=kinitializers.Constant(-(block_size - 1)),
        activation="relu",
    )(inp)
    flat = klayers.Flatten()(blockwise_sums)
    full_sum = klayers.Dense(
        1,
        kernel_initializer=kinitializers.Constant(-1),
        bias_initializer=kinitializers.Ones(),
        activation="relu",
    )(flat)
    out = klayers.Dense(
        1,
        kernel_initializer=kinitializers.Constant(-1),
        bias_initializer=kinitializers.Ones(),
        activation="linear",
    )(full_sum)
    return kmodels.Model(inp, out)


def create_scaled_kmodel(width, dimension=DIMENSION, block_size=BLOCK_SIZE):
    inp = klayers.Input(shape=(dimension, 1))
    shift_inp_layer = klayers.Conv1D(2, 1, strides=1, activation="relu")
    shifted_inp = shift_inp_layer(inp)
    shift_inp_layer.set_weights(
        [
            np.asarray([[[1, 1]]]),
            np.asarray([-0.5 + width / 2, -0.5 - width / 2]),
        ]
    )
    scale_inp_layer = klayers.Conv1D(1, 1, strides=1, activation="linear")
    scaled_inp = scale_inp_layer(shifted_inp)
    scale_inp_layer.set_weights(
        [np.asarray([[[1 / width], [-1 / width]]]), np.asarray([0])]
    )
    blockwise_sums = klayers.Conv1D(
        1,
        block_size,
        strides=1,
        kernel_initializer=kinitializers.Ones(),
        bias_initializer=kinitializers.Constant(-(block_size - 1)),
        activation="relu",
    )(scaled_inp)
    flat = klayers.Flatten()(blockwise_sums)
    full_sum = klayers.Dense(
        1,
        kernel_initializer=kinitializers.Constant(-1),
        bias_initializer=kinitializers.Ones(),
        activation="relu",
    )(flat)
    out = klayers.Dense(
        1,
        kernel_initializer=kinitializers.Constant(-1),
        bias_initializer=kinitializers.Ones(),
        activation="linear",
    )(full_sum)
    return kmodels.Model(inp, out)
