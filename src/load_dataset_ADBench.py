import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
import random
import pandas as pd

def fraud(dataset_path):
    N_train = 4000
    N_test_normal = 2000
    N_test_abnormal = 492
    data = np.load(dataset_path, allow_pickle=True)
    X, y = data['X'], data['y']
    X = MinMaxScaler().fit_transform(X)
    normal = X[np.where(y == 0)]
    anomalies = X[np.where(y == 1)]

    normal = shuffle(normal, random_state=1)
    train_normal = normal[:N_train]
    test_normal = normal[N_train:N_train+N_test_normal]
    test_abnormal = anomalies[:N_test_abnormal]

    x_test = np.concatenate((test_normal, test_abnormal), axis=0)
    y_test = np.concatenate(([1] * len(test_normal), [-1] * len(test_abnormal)), axis=0)
    y_train = []
    return train_normal, y_train, x_test, y_test
