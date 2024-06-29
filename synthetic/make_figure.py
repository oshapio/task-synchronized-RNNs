import glob
import os
import pickle
import numpy as np
import pandas as pd
import scipy

# load mackey glass sequence
full_seq = pickle.load(open(r"r_set_saved/rs_s=0.pkl", "rb"))
# take the middle sequence
take_id = int(len(full_seq["datas"]) / 2)
middle_seq = full_seq["datas"][take_id]

# path = r"./results_paper_simplified_v2_lorenz"
path = r"results_paper_simplified_v2_mackey_glass"
# read and aggregate and then aggregate again!

# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import seaborn as sns
import scienceplots

sns.reset_defaults()
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex"])
import numpy as np

# read and aggregate and then aggregate again!
# hide top and right splines
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
# also ticks
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

# hide minor ticks
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False

# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
font = {"weight": "normal", "size": 13}
plt.rc("font", **font)
# axes labels smaller size
plt.rc("xtick", labelsize=13)
plt.rc("ytick", labelsize=13)
plt.rc("axes", labelsize=13)


def read_folder(folder_path):
    os.chdir(folder_path)

    results = {}

    for file in reversed(sorted(glob.glob("*"))):
        if ".pkl" not in file:
            continue
        res = pickle.load(open(file, "rb"))
        if res["name"] not in results:
            results[res["name"]] = []
        results[res["name"]].append(res)
    return results


results = read_folder(path)

joint_df = pd.DataFrame()

for key, val in results.items():
    method_name = key
    for exp in val:
        # add in the results for each seed.
        exp_seed = exp["seed"]
        time_irregs = exp["r"]
        results_per_time = exp["results"]

        for time_id in range(len(time_irregs)):
            # if the method is vanilla exclude values > 1e2 and nans
            if results_per_time[time_id] > 1e8 or np.isnan(results_per_time[time_id]):
                continue
            joint_df = pd.concat(
                [
                    joint_df,
                    pd.DataFrame(
                        {
                            "method_name": method_name,
                            "seed": exp_seed,
                            "time_irregularity": time_irregs[time_id],
                            "mse": results_per_time[time_id],
                        },
                        index=[0],  # add a dummy index
                    ),
                ]
            )
fig, ax = plt.subplots(1, 1, figsize=(4, 3))


method_names = joint_df["method_name"].unique()
name_mapper = {
    "vanilla_esn": "ESN",
    "vanilla_esn_2_inputs": "ESN+$\Delta t$",
    "satesn": "TSESN",
}
colors = sns.color_palette("colorblind", 3)
for method_idx, method_name in enumerate(
    ["vanilla_esn", "vanilla_esn_2_inputs", "satesn"]
):
    df = joint_df[joint_df["method_name"] == method_name]

    sorted_time_irregs = np.sort(df["time_irregularity"].unique())

    mean_mse = df.groupby("time_irregularity")["mse"].mean()
    median_mse = df.groupby("time_irregularity")["mse"].median()
    std_mse = df.groupby("time_irregularity")["mse"].std()
    median_abs_dev = scipy.stats.median_abs_deviation(
        df.groupby("time_irregularity")["mse"].median()
    )
    min_mse = df.groupby("time_irregularity")["mse"].min()
    max_mse = df.groupby("time_irregularity")["mse"].max()
    # plot
    color = colors[method_idx]
    # ax.fill_between(sorted_time_irregs, mean_mse - std_mse, mean_mse + std_mse, edgecolor='gray', linewidth=5,
    #              facecolor='blue', alpha=0.2)
    ax.plot(
        sorted_time_irregs,
        median_mse,
        "-",
        label=name_mapper[method_name],
        markersize=5,
        linewidth=2,
        color=color,
    )
    ax.fill_between(
        sorted_time_irregs,
        min_mse,
        max_mse,
        edgecolor="white",
        linewidth=1,
        facecolor=color,
        alpha=0.1,
        # add cool effect
    )

ax.set_xlabel("Time irregularity $\pi$")
ax.set_ylabel("Test MSE")
# log y

ax.set_yscale("log")
# on;y show for mackey glass
if "mackey" in path:
    ax.legend(frameon=False)
# only set top limit
ax.set_ylim(top=1e2)
ax.set_xlim(left=-0.0001)
# make sure right xlim is covered in the labels and ticks
# ax.set_xticks([0, 0.01, 0.02, 0.028])
# ylim top: 1e4
# ax.set_ylim(top=1e1)
# save fig
exp_name = path.split("/")[-1]
fig.savefig(f"../{exp_name}-evaluation.pdf", bbox_inches="tight")
plt.show()
