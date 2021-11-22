import os

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# default parameters
if K.image_data_format() == "channels_first":
    INPUT_SHAPE = (1, 28, 28)
    IMAGE_SHAPE = INPUT_SHAPE[1:]
else:
    INPUT_SHAPE = (28, 28, 1)
    IMAGE_SHAPE = INPUT_SHAPE[:-1]

BATCH_SIZE_STAT = 10000  # batch size for statistics calculations
BATCH_SIZE_TRAIN = 128  # batch size for model training
BATCH_SIZE_TEST = 1  # batch size for model evaluation


# loader functions
def load_train_data(
    batch_size=BATCH_SIZE_STAT,
    class_mode="categorical",
    rescale=None,
    shuffle=False,
    samplewise_center=True,
):
    train_data_preprocessor = ImageDataGenerator(
        preprocessing_function=None,
        samplewise_center=samplewise_center,
        horizontal_flip=False,
        rescale=rescale,
    )
    train_generator = train_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "training"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="grayscale",
        shuffle=shuffle,
    )
    return train_generator


def load_train_data_augmented(
    batch_size=BATCH_SIZE_TRAIN,
    class_mode="categorical",
    rescale=None,
    shuffle=True,
    samplewise_center=True,
):
    train_data_preprocessor = ImageDataGenerator(
        preprocessing_function=None,
        samplewise_center=samplewise_center,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=False,
        rescale=rescale,
    )
    train_generator = train_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "training"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="grayscale",
        shuffle=shuffle,
    )
    return train_generator


def load_val_data(
    batch_size=BATCH_SIZE_TRAIN,
    class_mode="categorical",
    rescale=None,
    shuffle=False,
    samplewise_center=True,
):
    val_data_preprocessor = ImageDataGenerator(
        preprocessing_function=None,
        samplewise_center=samplewise_center,
        rescale=rescale,
    )
    val_generator = val_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "validation"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="grayscale",
        shuffle=shuffle,
    )
    return val_generator


def load_test_data(
    batch_size=BATCH_SIZE_TEST,
    class_mode=None,
    rescale=None,
    shuffle=False,
    samplewise_center=True,
):
    test_data_preprocessor = ImageDataGenerator(
        preprocessing_function=None,
        samplewise_center=samplewise_center,
        rescale=rescale,
    )
    test_generator = test_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "testing"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="grayscale",
        shuffle=shuffle,
    )
    return test_generator
