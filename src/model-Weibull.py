# Possion model
# %% import packages
import os
import numpy as np
from EQmodel import forecast
from multiprocessing import Pool


# %% specify training parameters
MCMC_param = dict(
    sample=dict(
        tune=20000,
        draws=10000,
        chains=4,
        cores=1,
        random_seed=123,
        # target_accept=0.9,
        nuts_sampler="pymc",
        idata_kwargs={"log_likelihood": True},
    ),
    sample_posterior_predictive=dict(
        extend_inferencedata=True,
        predictions=True,
        random_seed=123,
    ),
)


# %% import data
EQdata_path = "data/chronologies_all_final"
EQdata_name = [os.path.join(EQdata_path, name) for name in os.listdir(EQdata_path)]
EQdata_name.sort()
EQdata_param = dict(mc=100)

# %% transform data to interval
censor_year = 2022
EQdata_origin = [
    np.loadtxt(dm, delimiter=",", max_rows=EQdata_param["mc"]) for dm in EQdata_name
]
EQdata = [np.diff(data) for data in EQdata_origin]
EQdata_loo = [data[:, 0:-1] for data in EQdata]
EQdata_censor = [censor_year - data[:, -1] for data in EQdata_origin]

# %% define parallel function


def fore(i):
    forecast(
        i,
        dist="Weibull",
        parameters=MCMC_param,
        data=EQdata_loo[i].T,
        censored=False,
        data_censor=EQdata_censor[i].T,
    )


# %% main
if __name__ == "__main__":
    with Pool(processes=32) as pool:
        pool.map(fore, range(len(EQdata_name)))
