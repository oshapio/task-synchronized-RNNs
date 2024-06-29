# load train and testee
import pickle
import profile
import time

import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt

from utils.data_manager import DataManager

dm = DataManager()
d_train, d_test = dm.get_uwave_data(0.9)

valid_samples = int(0.3 * d_train[1].shape[1])
train_samples = d_train[1].shape[1] - valid_samples

# update valid set
d_valid = (
    d_train[0][-valid_samples:],
    d_train[1][:, -valid_samples:],
    d_train[2][-valid_samples:],
)

# update train set
d_train = (
    d_train[0][:train_samples],
    d_train[1][:, :train_samples],
    d_train[2][:train_samples],
)

# show labels distribution
count_train = np.sum(d_train[1], axis=1)
bc = np.linspace(1, 8, 8, endpoint=True)
plt.bar(bc, count_train)
dt_fun = lambda x: 1 - (2 ** (-x.astype(np.float32)))
train_x, train_y = d_train[0], d_train[1]
test_x, test_y = d_test[0], d_test[1]
valid_x, valid_y = d_valid[0], d_valid[1]
seq = np.arange(0, train_x.shape[1], 1)
stacked_x = np.vstack((seq, train_x)).T

seed = 1
resSize = 500

eigen_scaling = 1.26
a = 0.78
trainLen = d_train[0].shape[0]
reg = 0
testLen = 400
scaling = "locally"
model_name = "SATESN_{}res_backwardsdt_normdtmax_scaling_{}".format(resSize, scaling)

print("Starting `{}`".format(model_name))
# measure training time

# generate the ESN reservoir
inSize = outSize = 1

best_valid_accuracy = -1e9
parameters = []

np.random.seed(seed)

global_max_dt = 0

# compute global maximum dt
for index, sample in enumerate(train_x):
    # get time normalization var for this sample
    dts = np.array(d_train[2][index])
    dts = dts[1:] - dts[:-1]
    max_dt = np.max(dts)
    global_max_dt = max(max_dt, global_max_dt)
# compute global maximum dt
for index, sample in enumerate(valid_x):
    # get time normalization var for this sample
    dts = np.array(d_valid[2][index])
    dts = dts[1:] - dts[:-1]
    max_dt = np.max(dts)
    global_max_dt = max(max_dt, global_max_dt)
# compute global maximum dt
for index, sample in enumerate(test_x):
    # get time normalization var for this sample
    dts = np.array(d_test[2][index])
    dts = dts[1:] - dts[:-1]
    max_dt = np.max(dts)
    global_max_dt = max(max_dt, global_max_dt)


start_time = time.time()

print("Leaking - {}".format(a))

Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * 1
W = np.random.rand(resSize, resSize) - 0.5
# normalizing and setting spectral radius (correct, slow):
# print('Computing spectral radius...'),
rhoW = max(abs(linalg.eig(W)[0]))
# print('done.')
W *= eigen_scaling / rhoW

# allocated memory for the design (collected states) matrix
X = np.zeros((1 + inSize + resSize, trainLen))
# set the corresponding target matrix directly
Yt = d_train[1]
diff = []

# run the reservoir with the data and collect X
for index, sample in enumerate(train_x):
    # get time normalization var for this sample
    dts = np.array(d_train[2][index])
    dts = dts[1:] - dts[:-1]

    max_dt = np.max(dts)

    x = np.zeros((resSize, 1))
    for t in range(len(sample)):
        if t == 0:
            dt = 1
        else:
            if scaling == "globally":
                dt = global_max_dt
                raise Exception
            elif scaling == "locally":
                dt = dt_fun(dts[t - 1] / 10)
            else:
                print("Error: Method not implemented! Exiting.")
                exit(0)
        dt = min(dt, 1 / a)
        u = sample[t]
        x = (1 - a * dt) * x + a * dt * np.tanh(
            np.dot(Win, np.vstack((1, u))) + np.dot(W, x)
        )
    X[:, index] = np.vstack((1, u, x))[:, 0]

# train the output by ridge regression
X_T = X.T
Wout = np.dot(
    np.dot(Yt, X_T), linalg.inv(np.dot(X, X_T) + reg * np.eye(1 + inSize + resSize))
)

print("Training done. Took {} s".format(round(time.time() - start_time, 3)))

# vaidation
for index, sample in enumerate(valid_x):
    x = np.zeros((resSize, 1))

    dts = np.array(d_valid[2][index])
    dts = dts[1:] - dts[:-1]

    max_dt = np.max(dts)

    for t in range(len(sample)):
        if scaling == "globally":
            dt = dts[t - 1] / global_max_dt
            raise Exception
        elif scaling == "locally":
            dt = dt_fun(dts[t - 1] / 10)
        else:
            print("Error: Method not implemented! Exiting.")
            exit(0)
        dt = min(dt, 1 / a)

        u = sample[t]
        x = (1 - a * dt) * x + a * dt * np.tanh(
            np.dot(Win, np.vstack((1, u))) + np.dot(W, x)
        )
    y = np.dot(Wout, np.vstack((1, u, x)))
    # get best id
    b_id = np.argmax(y)

    diff.append(abs(b_id - np.argmax(valid_y[:, index])))
accuracy = np.count_nonzero(np.array(diff) == 0) / valid_y.shape[1]
print("Validation accuracy -> {}".format(accuracy))

# testing
diff = []
for index, sample in enumerate(test_x):
    x = np.zeros((resSize, 1))

    dts = np.array(d_test[2][index])
    dts = dts[1:] - dts[:-1]

    max_dt = np.max(dts)

    for t in range(len(sample)):
        if scaling == "globally":
            dt = global_max_dt
        elif scaling == "locally":
            dt = dt_fun(dts[t - 1] / 10)
        else:
            print("Error: Method not implemented! Exiting.")
            exit(0)
        dt = min(dt, 1 / a)

        u = sample[t]
        x = (1 - a * dt) * x + a * dt * np.tanh(
            np.dot(Win, np.vstack((1, u))) + np.dot(W, x)
        )
    y = np.dot(Wout, np.vstack((1, u, x)))
    # get best id
    b_id = np.argmax(y)

    diff.append(abs(b_id - np.argmax(test_y[:, index])))
accuracy = np.count_nonzero(np.array(diff) == 0) / test_y.shape[1]
print("Test accuracy -> {}".format(accuracy))
