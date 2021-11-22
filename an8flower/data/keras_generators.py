import os

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# default parameters
if K.image_data_format() == "channels_first":
    INPUT_SHAPE = (3, 128, 128)
    IMAGE_SHAPE = INPUT_SHAPE[1:]
else:
    INPUT_SHAPE = (128, 128, 3)
    IMAGE_SHAPE = INPUT_SHAPE[:-1]

BATCH_SIZE_STAT = 250  # batch size for statistics calculations
BATCH_SIZE_TRAIN = 64  # batch size for model training
BATCH_SIZE_TEST = 1  # batch size for model evaluation


VALIDATION_SPLIT = 0.1  # use 90% for training and 10% for validation


# loader functions
def load_train_data(batch_size=BATCH_SIZE_STAT, class_mode="categorical"):
    data_preprocessor = ImageDataGenerator(
        validation_split=VALIDATION_SPLIT, rescale=1.0 / 255.0,
    )
    train_generator = data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "an8Flower_double_12c"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=True,
        subset="training",
    )
    val_generator = data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "an8Flower_double_12c"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=False,
        subset="validation",
    )
    return train_generator, val_generator


def load_test_data(batch_size=BATCH_SIZE_TEST, class_mode=None):
    test_data_preprocessor = ImageDataGenerator(
        validation_split=VALIDATION_SPLIT, rescale=1.0 / 255.0,
    )
    test_generator = test_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "an8Flower_double_12c"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=False,
        subset="validation",
    )
    mask_generator = test_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "an8Flower_double_12c_MASK"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=False,
        subset="validation",
    )
    return test_generator, mask_generator
