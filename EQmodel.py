# %% import packages
import os
import numpy as np
import xarray as xr

# import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az

print(f"Running on PyMC v{pm.__version__}")

# %% create directory to save results
results_path = "results/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# %% specify training parameters
MCMC_param = dict(
    sample=dict(
        tune=10000,
        draws=10000,
        chains=4,
        cores=1,
        random_seed=10001,
        idata_kwargs={"log_likelihood": True},
    ),
    sample_posterior_predictive=dict(
        extend_inferencedata=True,
        predictions=True,
        random_seed=10001,
    ),
)

# %% import data
EQdata_path = "data/chronologies_all_final"
EQdata_name = [os.path.join(EQdata_path, name) for name in os.listdir(EQdata_path)]
EQdata_param = dict(mc=100)

# %% transform data to interval
censor_year = 2022
EQdata_origin = [
    np.loadtxt(dm, delimiter=",", max_rows=EQdata_param["mc"]) for dm in EQdata_name
]
EQdata = [np.diff(data) for data in EQdata_origin]
EQdata_censor = [censor_year - data[:, -1] for data in EQdata_origin]


class EQmodel:
    def __init__(
        self,
        data,
        data_censor=None,
        dist="Poisson",
        censored=False,
        verbose=True,
    ):
        self.data = data
        self.data_censor = data_censor
        self.dist = dist
        self.idata = None
        self.model = pm.Model()

        self.__init_model__(dist, censored, verbose)

    def __init_model__(self, dist, censored, verbose):

        match dist:
            case "Poisson":
                with self.model:
                    lam = pm.HalfNormal("lam", sigma=100)
                    sigmaZ = pm.HalfStudentT("sigmaZ", nu=3, sigma=5)
                    Z = pm.Gamma(
                        "Z",
                        alpha=1 / sigmaZ**2,
                        beta=1 / sigmaZ**2,
                        shape=(self.data.shape[1],),
                    )
                    interval = pm.Exponential("interval", lam * Z, observed=self.data)

            case "Gamma":
                with self.model:
                    alpha = pm.HalfNormal("alpha", sigma=100)
                    beta = pm.HalfNormal("beta", sigma=100)
                    sigmaZ = pm.HalfStudentT("sigmaZ", nu=3, sigma=5)
                    sigmaY = pm.HalfStudentT("sigmaY", nu=3, sigma=5)
                    Z = pm.Gamma(
                        "Z",
                        alpha=1 / sigmaZ**2,
                        beta=1 / sigmaZ**2,
                        shape=(self.data.shape[1],),
                    )
                    Y = pm.Gamma(
                        "Y",
                        alpha=1 / sigmaY**2,
                        beta=1 / sigmaY**2,
                        shape=(self.data.shape[1],),
                    )
                    interval = pm.Gamma(
                        "interval", alpha * Z, beta * Y, observed=self.data
                    )

            case "Lognormal":
                with self.model:
                    mu = pm.Normal("mu", mu=0, sigma=100)
                    sigma = pm.HalfStudentT("sigma", nu=3, sigma=5)
                    sigmaZ = pm.HalfStudentT("sigmaZ", nu=3, sigma=5)
                    sigmaY = pm.HalfStudentT("sigmaY", nu=3, sigma=5)
                    Z = pm.Normal("Z", mu=0, sigma=sigmaZ, shape=(self.data.shape[1],))
                    Y = pm.Gamma(
                        "Y",
                        alpha=1 / sigmaY**2,
                        beta=1 / sigmaZ**2,
                        shape=(self.data.shape[1],),
                    )
                    interval = pm.LogNormal(
                        "interval", mu=Z + mu, sigma=Y * sigma, observed=self.data
                    )

            case "Weibull":
                with self.model:
                    alpha = pm.HalfNormal("alpha", sigma=100)
                    beta = pm.HalfNormal("beta", sigma=100)
                    sigmaZ = pm.HalfStudentT("sigmaZ", nu=3, sigma=5)
                    sigmaY = pm.HalfStudentT("sigmaY", nu=3, sigma=5)
                    Z = pm.Gamma(
                        "Z",
                        alpha=1 / sigmaZ**2,
                        beta=1 / sigmaZ**2,
                        shape=(self.data.shape[1],),
                    )
                    Y = pm.Gamma(
                        "Y",
                        alpha=1 / sigmaY**2,
                        beta=1 / sigmaY**2,
                        shape=(self.data.shape[1],),
                    )
                    interval = pm.Weibull(
                        "interval",
                        alpha=Z * alpha,
                        beta=Y * beta,
                        observed=self.data,
                    )

            case "Wald":
                with self.model:
                    mu = pm.HalfNormal("mu", sigma=100)
                    alpha = pm.HalfStudentT("alpha", nu=3, sigma=5)
                    sigmaZ = pm.HalfStudentT("sigmaZ", nu=3, sigma=5)
                    sigmaY = pm.HalfStudentT("sigmaY", nu=3, sigma=5)
                    Z = pm.Gamma(
                        "Z",
                        alpha=1 / sigmaZ**2,
                        beta=1 / sigmaZ**2,
                        shape=(self.data.shape[1],),
                    )
                    Y = pm.Gamma(
                        "Y",
                        alpha=1 / sigmaY**2,
                        beta=1 / sigmaY**2,
                        shape=(self.data.shape[1],),
                    )
                    interval = pm.Wald(
                        "interval",
                        mu=1 / (Z * mu),
                        phi=Y * alpha,
                        observed=self.data,
                    )

            case _:
                raise NotImplementedError(
                    "The distribution " + dist + " is not implemented!"
                )

        if censored:
            # Add log probability for right censored data
            # Divide by times of occurrence to correct the log of likelihood
            pm.Potential(
                "censor",
                pt.log1mexp(pm.logcdf(interval, self.data_censor)) / self.data.shape[0],
                self.model,
            )

        if verbose:
            print("Initializing model using %s distribution!" % dist)
            # pm.model_to_graphviz(self.model).render(dist + "_model")

    def sample(self, parameters):

        with self.model:
            self.idata = pm.sample(**parameters)

    def sample_posterior_predictive(self, parameters):
        with self.model:
            pm.sample_posterior_predictive(self.idata, **parameters)

    def save(self, name, path=results_path):
        self.idata.to_netcdf(results_path + self.dist + "-" + name + ".nc")


poi_model = EQmodel(
    data=EQdata[0].T,
    data_censor=EQdata_censor[0].T,
    dist="Poisson",
    censored=True,
)
poi_model.sample(MCMC_param["sample"])
poi_model.sample_posterior_predictive(MCMC_param["sample_posterior_predictive"])

print(az.summary(poi_model.idata.predictions))
# poi_model.save(name="1")

# %% check convergence
# az.plot_trace(idata)
# az.summary(idata)
# az.loo(idata)
# az.waic(idata)

# %% predict variable interval
# with Poisson_Model:
#     pm.sample_posterior_predictive(idata,extend_inferencedata=True)