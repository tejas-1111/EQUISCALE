import numpy as np
import pandas as pd

# Changed to use data from EQUISCALE.


def get_adult():
    data_train = pd.read_csv(
        "../data/adult_train.csv", header=None, index_col=None
    ).to_numpy()
    data_test = pd.read_csv(
        "../data/adult_test.csv", header=None, index_col=None
    ).to_numpy()
    data = np.concatenate([data_train, data_test], axis=0)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def get_german():
    data = pd.read_csv("../data/german.csv", header=None, index_col=None).to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def get_compas():
    data = pd.read_csv("../data/compas.csv", header=None, index_col=None).to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def get_bank():
    data = pd.read_csv("../data/bank.csv", header=None, index_col=None).to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def get_default():
    data = pd.read_csv("../data/default.csv", header=None, index_col=None).to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    return X, y
