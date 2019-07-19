import os

import numpy as np
from keras.models import load_model, Model
from loguru import logger
from tqdm import tqdm

from src.data.shl_data import shl_min, shl_max, mean, std

# Define parameters

x_min = shl_min()[np.newaxis, np.newaxis, :]
x_max = shl_max()[np.newaxis, np.newaxis, :]
x_mean = mean()
x_std = std()

base = "../data/processed/X/hips"

logger.add("duplex_fe_cv.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
logger.info("Discriminator.")

sens_to_id = {"accel": 0, "gyro": 1, "mag": 2}
in_sensor = "gyro"
out_sensors = ["accel", "gyro"]

for out_sensor in out_sensors:
    sens_id = sens_to_id[out_sensor]
    for i in tqdm(range(5), desc="Folds", leave=False):
        # like this, because here we need EXACT hidden space representation
        model_name = f"{out_sensor}2{out_sensor}_duplex"

        tmp = f"../data/interim/hips/best_fold{i}_{model_name}_features"
        if not os.path.exists(tmp):
            os.makedirs(tmp)

        model = load_model(f"../models/hips/best_fold{i}_{model_name}")
        feature_encoder = Model(model.input, model.get_layer("features").output)

        for fname in tqdm(os.listdir(base), desc="files", leave=False):
            arr = np.load(base + "/" + fname)
            x = (arr - x_mean) / x_std
            x = (x - x_min) / (x_max - x_min)
            x = x[:, :, :, sens_id]
            x = x * 2 - 1

            features = feature_encoder.predict(x)
            np.save(f"{tmp}/{fname}", features)

        logger.info(f"{model_name} fold {i} finished!")
