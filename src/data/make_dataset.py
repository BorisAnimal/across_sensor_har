import os

import numpy as np
import pandas as pd
from keras.utils import to_categorical

from src.data.generators import create_generators
from src.data.shl_data import feature_columns, users, max_filename, min_filename, mean_filename, std_filename

dirname = os.path.dirname(__file__)

"""
This code extracts dataset.
In future folding will be done in generators.py file.
"""


def generate_dataframe(dir_path, mode="Hips"):
    """Generate csv from {mode}.txt with dir_path pointing to directory with file"""
    label_data = pd.DataFrame(np.loadtxt(f"{dir_path}/Label.txt"), columns=["Time(ms)", "Coarse", "Fine", "Road",
                                                                            "Traffic", "Tunnels", "Social", "Food"])
    data = pd.DataFrame(np.loadtxt(f"{dir_path}/{mode}_Motion.txt"), columns=["Time(ms)"] + feature_columns)

    assert (label_data.shape[0] == data.shape[0])
    data.set_index(pd.to_datetime(data['Time(ms)'], unit='ms'), inplace=True)
    label_data.set_index(pd.to_datetime(label_data['Time(ms)'], unit='ms'), inplace=True)
    data = pd.concat([data, label_data], join='inner', axis=1)
    data.drop(["Time(ms)", "ignore"], axis=1, inplace=True)
    data.dropna(inplace=True)
    del label_data
    return data


def raw_to_hdf():
    mode = "Hips"
    # for mode in body_locations:
    for user in users:
        _, subdirectories, _ = next(os.walk(os.path.abspath(f"raw/{user}/")))
        for subdirectory in subdirectories:
            print("Loading...", user, subdirectory)
            data = generate_dataframe(f"raw/{user}/{subdirectory}")
            kek = f"../../data/interim/{mode.lower()}_data/"
            if not os.path.exists(kek):
                os.makedirs(kek)
            data.to_hdf(f"../../data/interim/{mode.lower()}_data/{user}_{subdirectory}.hdf5", key="shl")


# TODO: pass subsampling parameter to go from 100Hz to another value
def data_sequencer(data: pd.DataFrame, sequence_length=500,
                   step=250, batch_size=128):
    """ Return generator producing sequences of fixed length from pd.DataFrame with some overlap
     in batches.
     """
    data.fillna(data.mean(), inplace=True)
    X = (data.iloc[:, np.r_[0:9, 10:19]])  # Select all columns except orient_w
    X = np.expand_dims(X, axis=2).reshape((X.shape[0], 3, 6))  # Stack sensors depthwise
    assert np.sum(X != X) == 0, f"Not cool - {np.sum(X != X)}, {X.shape}"

    Y = to_categorical(data['Fine'], num_classes=19)
    assert (X.shape[0] == Y.shape[0])
    sequence_starts = list(range(0, X.shape[0] - sequence_length, step))
    np.random.shuffle(sequence_starts)  # Shuffle sequences

    buffer_x = []
    buffer_y = []

    for index in sequence_starts:
        x = np.expand_dims(X[index:index + sequence_length], axis=0)
        y = np.expand_dims(Y[index + sequence_length], axis=0)

        buffer_x.append(x)
        buffer_y.append(y)

        if len(buffer_x) == batch_size:
            yield np.concatenate(buffer_x), np.concatenate(buffer_y)
            buffer_x.clear()
            buffer_y.clear()


def csv_to_npy_batches():
    for mode in ['hips']:
        for file in os.listdir(f"{dirname}/../../data/interim/{mode.lower()}_data/"):
            print(f"Uploading {file}!")
            data = pd.read_hdf(f"{dirname}/../../data/interim/{mode.lower()}_data/" + file, index_col="Time(ms)")
            for i, (X, Y) in enumerate(data_sequencer(data)):
                assert np.sum(X != X) == 0
                os.makedirs(f"{dirname}/../../data/processed/X/{mode.lower()}/", exist_ok=True)
                os.makedirs(f"{dirname}/../../data/processed/Y/{mode.lower()}/", exist_ok=True)
                np.save(f"{dirname}/../../data/processed/X/{mode.lower()}/{file}_{i}", X)
                np.save(f"{dirname}/../../data/processed/Y/{mode.lower()}/{file}_{i}", Y)


# Columns of initial dataset's tables
# Index(['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y',
#        'mag_z', 'orient_w', 'orient_x', 'orient_y', 'orient_z', 'grav_x',
#        'grav_y', 'grav_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'pressure',
#        'Coarse', 'Fine', 'Road', 'Traffic', 'Tunnels', 'Social', 'Food'],
#       dtype='object')

def generate_constants(mode="Hips"):
    base = f"{dirname}/../../data/interim/{mode.lower()}_data/"
    for file in os.listdir(f"{dirname}/../../data/interim/{mode.lower()}_data/"):
        mins = []
        maxs = []
        means = []
        stds = []
        lens = []

        for file in os.listdir(base):
            data = pd.read_hdf(os.path.join(base, file)).values
            mins.append(data.min(axis=0))
            maxs.append(data.max(axis=0))
            means.append(data.mean(axis=0))
            stds.append(data.std(axis=0))
            lens.append(len(data))
            del data

        np.save(min_filename, np.array(mins).min(axis=0)[np.r_[0:9, 10:19]].reshape(3, 6).min(axis=0))
        np.save(max_filename, np.array(maxs).max(axis=0)[np.r_[0:9, 10:19]].reshape(3, 6).max(axis=0))
        tmp = []
        for i in lens:
            tmp += [i] * 3

        l = np.array(tmp)

        pd.DataFrame((np.array(means) * l).sum(axis=0) / sum(lens)).to_csv(mean_filename, header=None)
        pd.DataFrame((np.array(stds) * l).sum(axis=0) / sum(lens)).to_csv(std_filename, header=None)




if not os.path.exists("../../data/interim"):
    print("Extracting raw dataset:")
    raw_to_hdf()
    generate_constants()

csv_to_npy_batches()

# To initialize all fold files split.
for i in range(5):
    kek = create_generators('Hips', f"s2s_fold{i}")
