import datetime
import os
import random
import sys
import time

import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as pp
import scipy as sp
import scipy.signal
from scipy.interpolate import interp1d
import pickle
import constants
import arff
import pandas as pd


class DataManager:
    readData = []

    def get_cave_data_dandak(self):
        df = pd.read_fwf("{}/data/other_cave/data.txt".format(constants.GLOBAL_PATH))[
            ::-1
        ]
        # return two columns
        df_np = df.values

        req_cols = df_np[:, :2]
        return req_cols

    def get_uwave_data_3d(
        self, missing_rate=0, interpolation_type=None, normalize_y=True
    ):
        """
        Returns UWaveGestureLibraryAll dataset as two tuples (Timestamps, Train, Test), where each of these containts training data as
        np array and labels array encoded into one-hot. Timestamps are indices of occurence
        :param missing_rate: Rate of missingness. Uniformly drops samples from the dataset
        :return: (train_timestamps, train_data, train_labels), (test_timestamps, test_data, test_labels)
        """  #
        path_cache = (
            constants.UWaveALLGestureDATASETPATH
            + "_3d/cache_{}_{}_{}.pkl".format(
                missing_rate, interpolation_type, normalize_y
            )
        )

        test_path, train_path = r"{}\UWaveGestureLibraryAll_TRAIN.arff".format(
            constants.UWaveALLGestureDATASETPATH
        ), r"{}\UWaveGestureLibraryAll_TEST.arff".format(
            constants.UWaveALLGestureDATASETPATH
        )
        # check if exists already
        if os.path.isfile(path_cache):
            print("Dataset is cached, loading from the disk..")
            dataset = pickle.load(open(path_cache, "rb"))
            print("Dataset Loaded!")
            return dataset

        # from scipy.io import arff
        import pandas as pd

        # data_train = arff.loadarff(train_path)
        # data_test  = arff.loadarff(test_path)

        print("Dataset is not cached, reading from scratch..")

        with open(train_path) as fh:
            cont = arff.load(fh)
        np_train = np.array(cont["data"]).astype(np.float32)
        data_train = np_train[:, :-1]
        train_3d = np.zeros((data_train.shape[0], data_train.shape[1] // 3, 3))
        for i in range(data_train.shape[0]):
            cnt = data_train.shape[1] // 3
            train_3d[i, :, 0] = data_train[i, 0:cnt]
            train_3d[i, :, 1] = data_train[i, cnt : cnt * 2]
            train_3d[i, :, 2] = data_train[i, cnt * 2 : cnt * 3]

        data_train_labels = np_train[:, -1].astype(np.int32)

        with open(test_path) as fh:
            cont = arff.load(fh)
        np_test = np.array(cont["data"]).astype(np.float32)
        data_test = np_test[:, :-1]
        data_test_labels = np_test[:, -1].astype(np.int32)

        # transform to 3d
        test_3d = np.zeros((data_test.shape[0], data_test.shape[1] // 3, 3))
        for i in range(data_test.shape[0]):
            cnt = data_test.shape[1] // 3
            test_3d[i, :, 0] = data_test[i, 0:cnt]
            test_3d[i, :, 1] = data_test[i, cnt : cnt * 2]
            test_3d[i, :, 2] = data_test[i, cnt * 2 : cnt * 3]

        # remove r*100% of every set
        new_train_x = []
        new_train_timestamps = []
        for index, i in enumerate(data_train):
            # take some random subset
            leave_in = train_3d[index, :].shape[0] - int(
                data_train[index, :].shape[0] * missing_rate
            )

            # donn't take the first and last datapoints
            sampled_ids = (
                [0]
                + sorted(
                    random.sample(
                        range(1, data_train[index].shape[0] - 1), leave_in - 2
                    )
                )
                + [data_train[index].shape[0] - 1]
            )

            dropped_train_seq = data_train[index][sampled_ids]
            if interpolation_type is not None:
                if interpolation_type == "linear":
                    print("Linear interpolation type found")
                    # do linear interpolation
                    f2 = interp1d(sampled_ids, dropped_train_seq, kind="linear")
                    sampled_ids = np.linspace(
                        0,
                        data_train[index].shape[0] - 1,
                        data_train[index].shape[0],
                        endpoint=True,
                    )
                    dropped_train_seq = f2(sampled_ids)
                else:
                    raise Exception(
                        "Error: Interpolation type is not implemented! Exiting."
                    )
            new_train_x.append(dropped_train_seq)
            new_train_timestamps.append(sampled_ids)

        # remove 10% of every test set
        new_test_x = []
        new_test_timestamps = []
        for index, i in enumerate(data_test):
            # take some random subset
            leave_in = data_test[index, :].shape[0] - int(
                data_test[index, :].shape[0] * missing_rate
            )
            sampled_ids = (
                [0]
                + sorted(
                    random.sample(range(1, data_test[index].shape[0] - 1), leave_in - 2)
                )
                + [data_test[index].shape[0] - 1]
            )

            dropped_test_seq = data_test[index][sampled_ids]

            if interpolation_type is not None:
                if interpolation_type == "linear":
                    print("Linear interpolation type found")
                    # do linear interpolation
                    f2 = interp1d(sampled_ids, dropped_test_seq, kind="linear")
                    sampled_ids = np.linspace(
                        0,
                        data_test[index].shape[0] - 1,
                        data_test[index].shape[0],
                        endpoint=True,
                    )
                    dropped_test_seq = f2(sampled_ids)
                else:
                    raise Exception(
                        "Error: Interpolation type is not implemented! Exiting."
                    )
            new_test_x.append(dropped_test_seq)
            new_test_timestamps.append(sampled_ids)

        new_train_x, new_test_x = np.array(new_train_x), np.array(new_test_x)

        train_labels_one_hot = np.eye(8)[data_train_labels - 1].T
        test_labels_one_hot = np.eye(8)[data_test_labels - 1].T

        # nornalize it?
        if normalize_y:
            min_pnt = min(new_train_x.min(), new_test_x.min())
            max_pnt = max(new_train_x.max(), new_test_x.max())

            new_train_x = (new_train_x - min_pnt) / (max_pnt - min_pnt)
            new_test_x = (new_test_x - min_pnt) / (max_pnt - min_pnt)

        print("Dataset read! Caching..")
        pickle.dump(
            [
                [new_train_x, train_labels_one_hot, new_train_timestamps],
                [new_test_x, test_labels_one_hot, new_test_timestamps],
            ],
            open(path_cache, "wb"),
        )
        print("Dataset saved!")
        return [new_train_x, train_labels_one_hot, new_train_timestamps], [
            new_test_x,
            test_labels_one_hot,
            new_test_timestamps,
        ]

    def get_uwave_data(
        self, missing_rate=0, interpolation_type=None, normalize_y=True, _3D=False
    ):
        """
        Returns UWaveGestureLibraryAll dataset as two tuples (Timestamps, Train, Test), where each of these containts training data as
        np array and labels array encoded into one-hot. Timestamps are indices of occurence
        :param missing_rate: Rate of missingness. Uniformly drops samples from the dataset
        :return: (train_timestamps, train_data, train_labels), (test_timestamps, test_data, test_labels)
        """
        path_cache = (
            constants.UWaveALLGestureDATASETPATH
            + "/cache_{}_{}_{}_3D_{}.pkl".format(
                missing_rate, interpolation_type, normalize_y, _3D
            )
        )

        test_path = r"{}/UWaveGestureLibraryAll_TRAIN.arff".format(
            constants.UWaveALLGestureDATASETPATH
        )
        train_path = r"{}/UWaveGestureLibraryAll_TEST.arff".format(
            constants.UWaveALLGestureDATASETPATH
        )
        # check if exists already
        if os.path.isfile(path_cache):
            print("Dataset is cached, loading from the disk..")
            dataset = pickle.load(open(path_cache, "rb"))
            print("Dataset Loaded!")
            return dataset

        # from scipy.io import arff
        import pandas as pd

        # data_train = arff.loadarff(train_path)
        # data_test  = arff.loadarff(test_path)

        print("Dataset is not cached, reading from scratch..")

        def convert_to_3D(univariate_seq):
            dim_len = univariate_seq.shape[1] // 3
            x, y, z = (
                univariate_seq[:, :dim_len],
                univariate_seq[:, dim_len : dim_len * 2],
                univariate_seq[:, dim_len * 2 :],
            )
            return np.stack((x, y, z), axis=-1)

        with open(train_path, "r") as fh:
            cont = arff.load(fh)
        attrs = cont["attributes"]
        np_train = np.array(cont["data"]).astype(np.float32)
        data_train_labels = np_train[:, -1].astype(np.int32)

        data_train = np_train[:, :-1]
        if _3D:
            data_train = convert_to_3D(data_train)

        with open(test_path) as fh:
            cont = arff.load(fh)
        np_test = np.array(cont["data"]).astype(np.float32)
        data_test = np_test[:, :-1]

        if _3D:
            data_test = convert_to_3D(data_test)

        data_test_labels = np_test[:, -1].astype(np.int32)

        # remove r*100% of every set
        new_train_x = []
        new_train_timestamps = []
        for index, i in enumerate(data_train):
            # take some random subset
            leave_in = data_train[index, :].shape[0] - int(
                data_train[index, :].shape[0] * missing_rate
            )

            # donn't take the first and last datapoints
            sampled_ids = (
                [0]
                + sorted(
                    random.sample(
                        range(1, data_train[index].shape[0] - 1), leave_in - 2
                    )
                )
                + [data_train[index].shape[0] - 1]
            )

            dropped_train_seq = data_train[index][sampled_ids]
            if interpolation_type is not None:
                if interpolation_type == "linear":
                    print("Linear interpolation type found")
                    # do linear interpolation
                    f2 = interp1d(sampled_ids, dropped_train_seq, kind="linear")
                    sampled_ids = np.linspace(
                        0,
                        data_train[index].shape[0] - 1,
                        data_train[index].shape[0],
                        endpoint=True,
                    )
                    dropped_train_seq = f2(sampled_ids)
                else:
                    raise Exception(
                        "Error: Interpolation type is not implemented! Exiting."
                    )
            new_train_x.append(dropped_train_seq)
            new_train_timestamps.append(sampled_ids)

        # remove 10% of every test set
        new_test_x = []
        new_test_timestamps = []
        for index, i in enumerate(data_test):
            # take some random subset
            leave_in = data_test[index, :].shape[0] - int(
                data_test[index, :].shape[0] * missing_rate
            )
            sampled_ids = (
                [0]
                + sorted(
                    random.sample(range(1, data_test[index].shape[0] - 1), leave_in - 2)
                )
                + [data_test[index].shape[0] - 1]
            )

            dropped_test_seq = data_test[index][sampled_ids]

            if interpolation_type is not None:
                if interpolation_type == "linear":
                    print("Linear interpolation type found")
                    # do linear interpolation
                    f2 = interp1d(sampled_ids, dropped_test_seq, kind="linear")
                    sampled_ids = np.linspace(
                        0,
                        data_test[index].shape[0] - 1,
                        data_test[index].shape[0],
                        endpoint=True,
                    )
                    dropped_test_seq = f2(sampled_ids)
                else:
                    raise Exception(
                        "Error: Interpolation type is not implemented! Exiting."
                    )
            new_test_x.append(dropped_test_seq)
            new_test_timestamps.append(sampled_ids)

        new_train_x, new_test_x = np.array(new_train_x), np.array(new_test_x)

        train_labels_one_hot = np.eye(8)[data_train_labels - 1].T
        test_labels_one_hot = np.eye(8)[data_test_labels - 1].T

        # nornalize it?
        if normalize_y:
            min_pnt = min(new_train_x.min(), new_test_x.min())
            max_pnt = max(new_train_x.max(), new_test_x.max())

            new_train_x = (new_train_x - min_pnt) / (max_pnt - min_pnt)
            new_test_x = (new_test_x - min_pnt) / (max_pnt - min_pnt)

        print("Dataset read! Caching..")
        pickle.dump(
            [
                [new_train_x, train_labels_one_hot, new_train_timestamps],
                [new_test_x, test_labels_one_hot, new_test_timestamps],
            ],
            open(path_cache, "wb"),
        )
        print("Dataset saved!")
        return [new_train_x, train_labels_one_hot, new_train_timestamps], [
            new_test_x,
            test_labels_one_hot,
            new_test_timestamps,
        ]
