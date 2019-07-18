import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Dense, Dropout, Flatten, Input)
from keras.models import Model
from loguru import logger

from src.data.generators import create_generators
from src.data.shl_data import mean, shl_max, shl_min, std

# from tqdm import tqdm

"""
This is K-Fold validation for DNN classifier
"""


def s_gen(generator):
    """
    Sensor Generator.
    Generator which takes batch from input and min-max normalize it
    :param generator:
    :return:
    """
    x_min = shl_min()[np.newaxis, np.newaxis, :]
    x_max = shl_max()[np.newaxis, np.newaxis, :]
    while True:
        x, y = next(generator)
        x = (x - x_min) / (x_max - x_min)
        x = x * 2 - 1  # Shift to (-1;1) interval
        s1_x = x[:, :, :, :3]  # Returns channels corresponding to inertia sensors
        yield s1_x, y


def get_encoder_dense(inp):
    """
    Get 1024-512-256 dense relu encoder
    :param inp:
    :return:
    """
    h = Flatten()(inp)
    h = Dense(1024, activation="relu")(h)
    h = Dense(512, activation="relu")(h)
    features = Dense(256, activation="relu", name="features")(h)
    h = Dropout(0.1)(features)
    return h


def get_classifier(inp):
    """128 - 19 dense classifiier"""
    h = Dense(128, activation="relu")(inp)
    h = Dense(19, activation="softmax")(h)
    return h


def get_callbacks(i, model_name):
    """
    Usefull callbacks used throguhout CV process
    :param i:
    :param model_name:
    :return:
    """
    es = EarlyStopping(patience=5)
    mc = ModelCheckpoint(f"models/hips/best_fold{i}_{model_name}", save_best_only=True)
    rlr = ReduceLROnPlateau(patience=3)
    return [es, mc, rlr]


def get_model():
    """
    Combines functions described above to construct a classifier.
    :return:
    """
    inp = Input(batch_shape=(128, 500, 3, 3))
    features = get_encoder_dense(inp)
    out_cls = get_classifier(features)
    model = Model(inp, out_cls)
    model.compile("rmsprop", "categorical_crossentropy", metrics=["accuracy"])
    return model


"""Useful precomputed constants"""
x_min = shl_min()[np.newaxis, np.newaxis, :]
x_max = shl_max()[np.newaxis, np.newaxis, :]
x_mean = mean()
x_std = std()

base = "data/interim/hips/data"

logger.add("classifier_1_k_cv.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
logger.info("CV base classifier have started.")


# Select sensors and their indices over which we will CV
# sensors = ["accel", "gyro", "mag"]
# sources = [0, 1, 2]

model_name = f"base_k_sensor"
# Five folds
for i in range(5):
    # Launch generators from filenames
    train_generator, test_generator = create_generators("hips", f"s2s_fold{i}")

    # # Filenames for each fold are stored in .npy. Load filenames.
    # train_fnames = np.load(f"data/filenames/s2s_fold{i}/train_filenames.npy")
    # val_fnames = np.load(f"data/filenames/s2s_fold{i}/val_filenames.npy")



    # Min max scaling
    train_gen, test_gen = s_gen(train_generator), s_gen(test_generator)
    model = get_model()
    logger.info(f"Processing {model_name}_fold_{i}...")
    history = model.fit_generator(train_gen, steps_per_epoch=748, epochs=300,
                                  callbacks=get_callbacks(i, model_name), verbose=1,
                                  validation_data=test_gen, validation_steps=187)

    history_data = np.array([history.history['val_acc'],
                             history.history['val_loss'],
                             history.history['acc'],
                             history.history['loss']])

    np.save(f"histories/{model_name}_fold_{i}", history_data)

    logger.info(f"{model_name} finished")