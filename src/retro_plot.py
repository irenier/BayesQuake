import numpy as np
import pickle
import pandas as pd
import matplotlib
import os

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# %% load data
results_path = "./retro_results/"
summary_path = results_path + "summary/"
figures_path = results_path + "figures/"

if not os.path.exists(figures_path):
    os.makedirs(figures_path)

EQdata_path = "data/chronologies_all_final"
EQdata_name = [os.path.join(EQdata_path, name) for name in os.listdir(EQdata_path)]
EQdata_name.sort()
EQdata_param = dict(mc=100)

EQdata_origin = [
    np.loadtxt(dm, delimiter=",", max_rows=EQdata_param["mc"]) for dm in EQdata_name
]
EQdata = [np.diff(data) for data in EQdata_origin]
EQdata_last = [data[:, -1] for data in EQdata]

# %% set parameters
N = len(EQdata_origin)
# N = 45
model_name = ["Poisson", "MA_waic"]
model_color = ["gray", "steelblue"]
tol_wd = 0.4

# %% statistics
mse = np.zeros((len(model_name),))
recall = np.zeros((len(model_name),))

# %% plot retrospective CI
fig, ax = plt.subplots(figsize=(40, 10))
ax.set_yscale("symlog", linthresh=5e2, linscale=1.5)
ax.set_xticks(np.arange(1, N + 1) + tol_wd / 2)
ax.set_xticklabels([str(i) for i in np.arange(1, N + 1)])
ax.set_xlabel("fault")
ax.set_ylabel("year")
ax.set_ylim(-1e7, 1e8)

for i in range(N):
    with open(summary_path + str(i + 1) + ".pkl", "rb") as f:
        df_raw = pickle.load(f)
    df = df_raw.loc[model_name]

    last_occ = np.mean(EQdata_last[i])
    quant = np.array(list(df["quantile"])) - last_occ
    mean = np.array(list(df["mean"])) - last_occ

    # compute mse
    mse += mean**2

    # compute recall rate
    recall += np.prod(np.sign(quant), axis=1) == -1.0

    ax.vlines(
        np.linspace(i + 1, i + 1 + tol_wd, df.shape[0]),
        quant[:, 0],
        quant[:, 1],
        colors=model_color,
    )
    ax.scatter(
        np.linspace(i + 1, i + 1 + tol_wd, df.shape[0]),
        mean,
        s=10,
        c=model_color,
    )

ax.axhline(0, color="blue")

mypatch = [
    mlines.Line2D([], [], color=j, label=i)
    for (i, j) in zip(
        model_name,
        model_color,
    )
]
# mypatch = mypatch.append(ax.scatter([], [], color="black", label="mean"))
ax.legend(handles=mypatch)
fig.savefig(figures_path + "retro_forecast.pdf")

mse = mse / N
rmse = np.sqrt(mse)
nrmse = rmse / np.mean(np.array(EQdata_last))

data = pd.DataFrame(
    index=model_name,
    columns=["MSE", "RMSE", "NRMSE", "Recall", "Recall rate"],
    data=np.array([mse, rmse, nrmse, recall, recall / N]).T,
)

data.to_csv(results_path + "mse_recall.csv")
