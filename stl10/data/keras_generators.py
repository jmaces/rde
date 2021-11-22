import os

from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# default parameters
if K.image_data_format() == "channels_first":
    INPUT_SHAPE = (3, 224, 224)
    IMAGE_SHAPE = INPUT_SHAPE[1:]
else:
    INPUT_SHAPE = (224, 224, 3)
    IMAGE_SHAPE = INPUT_SHAPE[:-1]

BATCH_SIZE_STAT = 250  # batch size for statistics calculations
BATCH_SIZE_TRAIN = 64  # batch size for model training
BATCH_SIZE_TEST = 1  # batch size for model evaluation


# loader functions
def load_train_data(batch_size=BATCH_SIZE_STAT, class_mode="categorical"):
    train_data_preprocessor = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        samplewise_center=False,
        horizontal_flip=False,
    )
    train_generator = train_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "training"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=False,
    )
    return train_generator


def load_train_data_augmented(
    batch_size=BATCH_SIZE_TRAIN, class_mode="categorical"
):
    train_data_preprocessor = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        samplewise_center=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    train_generator = train_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "training"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="rgb",
    )
    return train_generator


def load_val_data(batch_size=BATCH_SIZE_TRAIN, class_mode="categorical"):
    val_data_preprocessor = ImageDataGenerator(
        preprocessing_function=preprocess_input, samplewise_center=False,
    )
    val_generator = val_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "validation"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=False,
    )
    return val_generator


def load_test_data(batch_size=BATCH_SIZE_TEST, class_mode=None):
    test_data_preprocessor = ImageDataGenerator(
        preprocessing_function=preprocess_input, samplewise_center=False,
    )
    test_generator = test_data_preprocessor.flow_from_directory(
        os.path.join(os.path.split(__file__)[0], "testing"),
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=False,
    )
    return test_generator
