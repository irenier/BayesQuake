import numpy as np
import arviz as az
import pandas as pd
import pickle
import os
from multiprocessing import Pool

# %% load data
EQdata_path = "data/chronologies_all_final"
EQdata_name = [os.path.join(EQdata_path, name) for name in os.listdir(EQdata_path)]
EQdata_param = dict(mc=100)

with open("data/EQname.pkl", "wb") as f:
    pickle.dump(EQdata_name, f)

EQdata_origin = [
    np.loadtxt(dm, delimiter=",", max_rows=EQdata_param["mc"]) for dm in EQdata_name
]
EQdata = [np.diff(data) for data in EQdata_origin]
EQdata_loo = [data[:, 0:-1] for data in EQdata]

# %% specify parameters

results_path = "./retro_results/"
summary_path = results_path + "summary/"

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

dist_name = ["Poisson", "Gamma", "Lognormal", "Weibull", "Wald"]
quantile_alpha = 0.05
quantile_CI = [quantile_alpha / 2, 1 - quantile_alpha / 2]
weight_exp_beta = 1.0


def summary(i):
    res_model = dict()
    res_compare = dict()
    pred = list()
    for j in range(5):
        dist_name_j = dist_name[j]
        file_name = results_path + dist_name_j + "-" + str(i + 1) + ".nc"
        model = az.from_netcdf(file_name)
        res_compare[dist_name_j] = model
        pred.append(np.array(model.predictions.interval))
        res_model[dist_name_j] = dict(
            median=np.median(pred[j]),
            mean=np.mean(pred[j]),
            quantile=np.quantile(pred[j], quantile_CI),
            waic=az.waic(model).elpd_waic,
            loo=az.loo(model).elpd_loo,
        )

    pred_calc = np.array(pred)

    res_df = pd.DataFrame.from_dict(res_model, orient="index")
    res_df["weight_waic"] = np.exp(res_df.waic - np.max(res_df.waic)) / np.sum(
        np.exp(res_df.waic - np.max(res_df.waic))
    )
    res_df["weight_loo"] = az.compare(res_compare).weight

    MA_pred = dict(
        MA_waic=np.einsum("a,abcde->bcde", np.array(res_df["weight_waic"]), pred_calc),
        MA_loo=np.einsum("a,abcde->bcde", np.array(res_df["weight_loo"]), pred_calc),
    )

    MA_model = dict()
    for j in ["MA_waic", "MA_loo"]:
        MA_model[j] = dict(
            median=np.median(MA_pred[j]),
            mean=np.mean(MA_pred[j]),
            quantile=np.quantile(MA_pred[j], quantile_CI),
        )
    MA_df = pd.DataFrame.from_dict(MA_model, orient="index")

    with open(summary_path + str(i + 1) + ".pkl", "wb") as f:
        pickle.dump(pd.concat([res_df, MA_df]), f)


# %% main
if __name__ == "__main__":
    with Pool(processes=24) as pool:
        pool.map(summary, range(len(EQdata_name)))
