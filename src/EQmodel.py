# %% import packages
import os
import arviz as az
import pymc as pm
import pytensor.tensor as pt

# %% create directory to save results
results_path = "results/"
if not os.path.exists(results_path):
    os.makedirs(results_path)


# %% define model class
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
                    interval = pm.Exponential(
                        "interval", 1 / (lam * Z), observed=self.data
                    )

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
                        "interval", 1 / (alpha * Z), 1 / (beta * Y), observed=self.data
                    )

            case "Lognormal":
                with self.model:
                    mu = pm.HalfNormal("mu", sigma=100)
                    sigma = pm.HalfStudentT("sigma", nu=3, sigma=5)
                    sigmaZ = pm.HalfStudentT("sigmaZ", nu=3, sigma=5)
                    sigmaY = pm.HalfStudentT("sigmaY", nu=3, sigma=5)
                    Z = pm.HalfNormal("Z", sigma=sigmaZ, shape=(self.data.shape[1],))
                    Y = pm.Gamma(
                        "Y",
                        alpha=1 / sigmaY**2,
                        beta=1 / sigmaY**2,
                        shape=(self.data.shape[1],),
                    )
                    interval = pm.LogNormal(
                        "interval", mu=Z + mu, tau=1 / (Y * sigma), observed=self.data
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
                        alpha=1 / (Z * alpha),
                        beta=1 / (Y * beta),
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
                        lam=Y * alpha,
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

    def save(self, name, path):
        self.idata.to_netcdf(
            path + self.dist + "-" + name + ".nc", overwrite_existing=True
        )


# %% construct model to fit
def forecast(i, dist, parameters, data, data_censor=None, verbose=True):

    if verbose:
        print("Processing data %s!" % (i + 1))

    model = EQmodel(data=data, data_censor=data_censor, dist=dist, censored=False)
    model.sample(parameters["sample"])
    model.sample_posterior_predictive(parameters["sample_posterior_predictive"])
    model.save(name=str(i + 1), path=results_path)

    # print(az.summary(model.idata.predictions))

    # # check convergence
    # az.plot_trace(model.idata)
    # az.summary(model.idata)
