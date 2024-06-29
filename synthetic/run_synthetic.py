""" Based on implementation by Mantas Lukoševičius (http://mantas.info) """

import random

import numpy as np
from matplotlib.pyplot import *
import scipy.linalg
import collections
from mpl_toolkits.mplot3d import Axes3D
from numpy import zeros, dot, vstack, eye
from numpy.linalg import linalg
from scipy.interpolate import interp1d
import pickle

import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Mackey Glass experiment")
parser.add_argument(
    "--model_type",
    type=str,
    default="satesn",
    help="Model type to run",
    choices=["vanilla_esn_2_inputs", "tsesn", "vanilla_esn"],
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mackey_glass",
    help="Dataset to run",
    choices=["mackey_glass", "lorenz"],
)

args = parser.parse_args()
model_type = args.model_type
dataset = args.dataset

print(f"Running model type: {model_type} on dataset: {dataset}")


def interpolated_mackey_glass(sample_len=1000, tau=17, seed=None, sample_r=0):
    """
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    """
    delta_t = 100
    history_len = tau * delta_t
    # Initial conditions for the history of the system
    timeseries = 1.2

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # sample the sequence with very high resolution
    const_sampled_t = []
    current_sample = 0

    history = collections.deque(
        1.2 * np.ones(history_len) + 0.2 * (np.random.rand(history_len) - 0.5)
    )
    # Preallocate the array for the time-series
    inp = np.zeros((sample_len * 100, 1))

    for timestep in range(sample_len):
        dt = 1

        step_size = dt / delta_t
        for _ in range(delta_t):
            xtau = history.popleft()
            history.append(timeseries)
            timeseries = (
                history[-1]
                + (0.2 * xtau / (1.0 + xtau**10) - 0.1 * history[-1]) * step_size
            )
            const_sampled_t.append(step_size)
            inp[current_sample] = timeseries
            current_sample += 1

    inp = np.tanh(inp - 1)

    # optional
    for i in range(1, len(const_sampled_t)):
        const_sampled_t[i] += round(const_sampled_t[i - 1], 5)
        const_sampled_t[i] = round(const_sampled_t[i], 5)

    f2 = interp1d(const_sampled_t, inp.flatten(), kind="cubic")

    # now sample according to the range
    actual_sequence = np.zeros((sample_len))
    timesteps = []
    timesteps.append(const_sampled_t[0])
    actual_sequence[0] = f2(timesteps[0])

    for i in range(1, sample_len):
        dt = 1 + (random.random() - 0.5) * 2 * sample_r
        timesteps.append(round(timesteps[-1] + dt, 5))
        actual_sequence[i] = f2(min(timesteps[-1], const_sampled_t[-1]))
    return actual_sequence, np.array(timesteps)


def lorenz(sample_len=1000, sigma=10, rho=28, beta=8 / 3, step=0.01, sample_r=0):
    """This function generates a Lorentz time series of length sample_len,
    with standard parameters sigma, rho and beta.
    """

    x = np.zeros([sample_len])
    y = np.zeros([sample_len])
    z = np.zeros([sample_len])

    timestamp = np.zeros((sample_len))

    # Initial conditions taken from 'Chaos and Time Series Analysis', J. Sprott
    x[0] = 0
    y[0] = -0.01
    z[0] = 9

    for t in range(sample_len - 1):
        # pick some dt with mean of 0.01
        dt_step = -1
        while dt_step <= 0:
            dt_step = 0.01 + (random.random() - 0.5) * 2 * sample_r

        # subsample it
        x[t + 1] = x[t] + sigma * (y[t] - x[t]) * dt_step
        y[t + 1] = y[t] + (x[t] * (rho - z[t]) - y[t]) * dt_step
        z[t + 1] = z[t] + (x[t] * y[t] - beta * z[t]) * dt_step

        timestamp[t + 1] = timestamp[t] + dt_step
    x.shape += (1,)
    y.shape += (1,)
    z.shape += (1,)

    return np.concatenate((x, y, z), axis=1), timestamp


if dataset == "mackey_glass":
    # load the data
    trainLen = 3000
    testLen = 2000
    initLen = 500

    inSize = outSize = 1

    frome, to, step = 0, 1, 0.01
    iter_range = np.arange(frome, to, step)
    resSize = 500
    a = 0.4
    eigen_scaling = 1
    reg = 1e-7

    errorLen = 200
    if not os.path.exists(r"r_set_saved"):
        os.makedirs(r"r_set_saved")
    if not os.path.exists(r"r_set_saved/rs_s=10.pkl"):
        print("pre-calculating r set for 50 seeds")
        for seed in tqdm(range(50)):
            datas = []
            rs = []
            # check if saved
            for r in iter_range:
                data, delta = interpolated_mackey_glass(6000, seed=seed, sample_r=r)
                datas.append((data, delta))
                rs.append(r)

            # save r for the paper
            pickle.dump(
                {"rs": rs, "datas": datas}, open(f"r_set_saved/rs_s={seed}.pkl", "wb")
            )

    else:
        print("Loading r set from disk")
        # data = pickle.load(open(r"r_set_saved/rs.pkl", "rb"))
        # datas = data["datas"]
        # rs = data["rs"]
        data_seed = [
            pickle.load(open(f"r_set_saved/rs_s={seed}.pkl", "rb"))
            for seed in range(10)
        ]
        datas_seed = [data["datas"] for data in data_seed]
        rs_seed = [data["rs"] for data in data_seed]

        # make sure data[0] is of (N, 1) shape
        for i in range(len(datas_seed)):
            for j in range(len(datas_seed[i])):
                datas_seed[i][j] = (
                    datas_seed[i][j][0].reshape(-1, 1),
                    datas_seed[i][j][1],
                )
            # datas[i] = (datas[i][0].reshape(-1, 1), datas[i][1])
elif dataset == "lorenz":
    # load the data
    trainLen = 3000
    testLen = 2000
    initLen = 500

    inSize = outSize = 3

    frome, to, step = 0, 0.03, 0.001
    iter_range = np.arange(frome, to, step)
    rs = iter_range
    resSize = 500
    a = 0.30000000000000004  # leaking rate
    eigen_scaling = 0.2
    reg = 1e-7
    errorLen = 200

    def get_data_with_seed(seed):
        datas = [lorenz(6000, sample_r=r) for r in iter_range]
        # normalize the deltas s.t. the mean is 1.0
        # also norm sequence to be normed in [0, 1]
        for i in range(len(datas)):
            data_i = datas[i]
            ts = data_i[1]
            deltas = ts[1:] - ts[:-1]
            mean_delta = np.mean(deltas)

            ts_normed = ts / mean_delta

            norm_seq = (data_i[0] - data_i[0].min(axis=0)) / (
                data_i[0].max(axis=0) - data_i[0].min(axis=0)
            )
            datas[i] = (norm_seq, ts_normed)
        return datas


def saver(name, rs, seed, results, predictions=None):
    os.makedirs(r"./results_paper_simplified_v2_{}/".format(dataset), exist_ok=True)
    pickle.dump(
        {
            "name": name,
            "seed": seed,
            "results": results,
            "error_len": errorLen,
            "reg": reg,
            "r": rs,
            "res_size": resSize,
            "predictions": predictions,
            "leaking_rate": a,
            "eigen_scaling": eigen_scaling,
        },
        open(
            r"./results_paper_simplified_v2_{}/{}-{}-{}-{}-{}-{}.pkl".format(
                dataset, name, len(rs), seed, a, eigen_scaling, reg
            ),
            "wb",
        ),
    )


inSizeOrg = inSize
one_vec = np.array([[1.0]])
seeds = [i for i in range(0, 10)]
for current_seed in tqdm(seeds):
    print(f"Running seed {current_seed}")
    errors_model = []
    if dataset == "lorenz":
        # regen data with seed
        datas = get_data_with_seed(current_seed)
    elif dataset == "mackey_glass":
        datas = datas_seed[current_seed]
        rs = rs_seed[current_seed]

    # save results after every run
    for data, deltas in datas:
        # generate the ESN reservoir
        if model_type == "vanilla_esn" or model_type == "satesn":
            inSize = inSizeOrg
        else:
            inSize = inSizeOrg + 1

        random.seed(current_seed)
        np.random.seed(current_seed)
        Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * 1
        W = np.random.rand(resSize, resSize) - 0.5
        # normalizing and setting spectral radius (correct, slow):
        rhoW = max(abs(linalg.eig(W)[0]))
        W *= eigen_scaling / rhoW

        # allocated memory for the design (collected states) matrix
        X = zeros((1 + inSize + resSize, trainLen - initLen))
        # set the corresponding target matrix directly
        Yt = data[initLen + 1 : trainLen + 1].T

        # run the reservoir with the data and collect X
        x = zeros((resSize, 1))
        for t in range(trainLen):
            u = data[t].reshape(-1, 1)
            dt = (deltas[t + 1] - deltas[t]).item()
            if model_type == "vanilla_esn":
                x = (1 - a) * x + a * np.tanh(dot(Win, vstack((1, u))) + dot(W, x))
            elif model_type == "vanilla_esn_2_inputs":
                x = (1 - a) * x + a * np.tanh(dot(Win, vstack((1, u, dt))) + dot(W, x))
            elif model_type == "tsesn":
                dt = min(1 / a, dt)
                x = (1 - a * dt) * x + a * dt * np.tanh(
                    dot(Win, vstack((1, u))) + dot(W, x)
                )
            if t >= initLen:
                if model_type == "vanilla_esn":
                    X[:, t - initLen] = vstack((1, u, x))[:, 0]
                elif model_type == "vanilla_esn_2_inputs":
                    X[:, t - initLen] = vstack((1, u, dt, x))[:, 0]
                elif model_type == "tsesn":
                    X[:, t - initLen] = vstack((1, u, x))[:, 0]

        # train the output by ridge regression
        X_T = X.T
        Wout = dot(
            dot(Yt, X_T), linalg.inv(dot(X, X_T) + reg * eye(1 + inSize + resSize))
        )

        # run the trained ESN in a generative mode. no need to initialize here,
        # because x is initialized with training data and we continue from there.
        Y = zeros((outSize, testLen))
        u = data[trainLen].reshape(-1, 1)
        for t in range(testLen):
            dt = (deltas[trainLen + t + 1] - deltas[trainLen + t]).item()
            if model_type == "vanilla_esn":
                x = (1 - a) * x + a * np.tanh(dot(Win, vstack((1, u))) + dot(W, x))
                y = dot(Wout, vstack((1, u, x)))
            elif model_type == "vanilla_esn_2_inputs":
                x = (1 - a) * x + a * np.tanh(
                    dot(Win, vstack((one_vec, u, dt))) + dot(W, x)
                )
                y = dot(Wout, vstack((1, u, dt, x)))
            elif model_type == "tsesn":
                x = (1 - a * dt) * x + dt * a * np.tanh(
                    dot(Win, vstack((one_vec, u))) + dot(W, x)
                )
                y = dot(Wout, vstack((one_vec, u, x)))

            Y[:, t] = y[:, 0]
            # generative mode:
            u = y
            ## this would be a predictive mode:
            # u = data[trainLen+t+1]

        # compute MSE for the first errorLen time steps

        mse = (
            np.sum(
                np.square(
                    data[trainLen + 1 : trainLen + errorLen + 1] - Y[:, 0:errorLen].T
                )
            )
            / errorLen
        )
        errors_model.append(mse)
    saver(model_type, rs, current_seed, errors_model)
