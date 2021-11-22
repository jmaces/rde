import numpy as np


# GLOBAL DEFAULT PARAMETERS
DIMENSION = 16
BLOCK_SIZE = 5


# INSTANCES WITH BLOCK
def create_zero_bg_block(
    dimension=DIMENSION, block_size=BLOCK_SIZE, position=None
):
    if position is None:
        position = np.random.randint(0, dimension - block_size + 1)
    assert position + block_size <= dimension
    bg = np.zeros(dimension)
    bg[position : position + block_size] = 1
    return bg, (position, position + block_size)


def create_alternating_bg_block(
    dimension=DIMENSION,
    block_size=BLOCK_SIZE,
    position=None,
    skip=2,
    shift=None,
):
    if position is None:
        position = np.random.randint(0, dimension - block_size + 1)
    if shift is None:
        shift = np.random.randint(0, skip)
    assert position + block_size <= dimension
    assert shift < skip
    bg = np.zeros(dimension)
    bg[shift::skip] = 1
    bg[position : position + block_size] = 1
    return bg, (position, position + block_size)


def create_bernoulli_bg_block(
    dimension=DIMENSION, block_size=BLOCK_SIZE, position=None, p=0.1
):
    if position is None:
        position = np.random.randint(0, dimension - block_size + 1)
    assert position + block_size <= dimension
    bg = np.random.binomial(1, p, dimension)
    bg[position : position + block_size] = 1
    return bg, (position, position + block_size)


def create_almost_block_bg_block(dimension=DIMENSION, block_size=BLOCK_SIZE):
    assert dimension >= 2 * block_size + 5
    assert block_size >= 3
    bg = np.zeros(dimension)
    bg[1 : 1 + block_size // 2] = 1
    bg[2 + block_size // 2 : 1 + block_size] = 1
    bg[-1 - block_size : -1] = 1
    bg[2 + block_size + (dimension - 4 - 2 * block_size) // 2] = 1
    return bg, (dimension - 1 - block_size, dimension - 1)


# INSTANCES WITHOUT BLOCK
def create_zero_bg(dimension=DIMENSION):
    bg = np.zeros(dimension)
    return bg


def create_alternating_bg(dimension=DIMENSION, skip=2, shift=None):
    if shift is None:
        shift = np.random.randint(0, skip)
    assert shift < skip
    bg = np.zeros(dimension)
    bg[shift::skip] = 1
    return bg


def create_bernoulli_bg(dimension=DIMENSION, p=0.1):
    bg = np.random.binomial(1, p, dimension)
    return bg
