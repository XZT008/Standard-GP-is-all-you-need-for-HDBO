import numpy as np
import torch
import gpytorch
import math
from data import *
import random
from tqdm import tqdm
import ssl
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.gpytorch import GPyTorchModel
from torch.quasirandom import SobolEngine
from gpytorch.constraints import Positive
from gpytorch.priors import HalfCauchyPrior, LogNormalPrior, GammaPrior, UniformPrior, MultivariateNormalPrior, NormalPrior
from gpytorch.functions import MaternCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels.kernel import Kernel
from typing import Optional
import pickle
from infras.randutils import *
from benchmark.rover_function import Rover
from benchmark.naslib_benchmark import NasBench201
from benchmark.svm_benchmark import SVMBenchmark
from benchmark.mopta8 import MoptaSoftConstraints
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from pyro.infer.mcmc import NUTS, MCMC
import pyro


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class Saas_MaternKernel(Kernel):
    has_lengthscale = True

    def __init__(self, nu: Optional[float] = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(Saas_MaternKernel, self).__init__(**kwargs)
        self.nu = nu

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).mul(self.lengthscale)
            x2_ = (x2 - mean).mul(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, 1.0/self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )


class SaasGP_MAP(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, tau: float):
        super(SaasGP_MAP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            Saas_MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_prior=HalfCauchyPrior(scale=tau)),
            outputscale_prior=LogNormalPrior(loc=0.0, scale=10.0),
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SaasGP_MAP_Wrapper:
    def __init__(self, train_x, train_y):
        self.X = train_x
        self.y = train_y.squeeze()
        self.taus = [1.0, 1e-1, 1e-2, 1e-3]
        self.likelihoods = [gpytorch.likelihoods.GaussianLikelihood() for i in range(4)]
        self.gp_models = [SaasGP_MAP(self.X, self.y, self.likelihoods[i], self.taus[i]) for i in range(4)]
        self.gp_model = None
        self.likelihood = None

    def train_model(self, epochs=1500, lr=0.02):
        out_logprob = []

        # train all models
        for i in range(4):
            self.gp_models[i].train()
            self.likelihoods[i].train()
            optimizer = torch.optim.Adam(self.gp_models[i].parameters(), lr=lr, betas=(0.50, 0.999))
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[i], self.gp_models[i])

            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self.gp_models[i](self.X)
                # Calc loss and backprop gradients
                loss = -mll(output, self.y)
                loss.backward()
                optimizer.step()

            # select tau
            # create a new gp model for each leave one out
            log_prob = 0.0
            mean_mat = self.gp_models[i].mean_module(self.X)
            const_mean = mean_mat[0]
            covar_mat = self.gp_models[i].covar_module(self.X).evaluate()
            noise = self.likelihoods[i].noise
            for j in range(self.X.shape[0]):
                train_y = torch.cat([self.y[:j], self.y[j + 1:]])
                test_y = self.y[j:j + 1]

                Kxx = torch.cat([covar_mat[:j], covar_mat[j+1:]])
                Kxx = torch.cat([Kxx[:, :j], Kxx[:, j+1:]], dim=1)
                Kx = covar_mat[j:j+1]
                Kx = torch.cat([Kx[:, :j], Kx[:, j+1:]], dim=1)
                k = covar_mat[j, j]
                noise_mat = torch.eye(self.X.shape[0]-1) * noise

                mean = torch.matmul(torch.matmul(Kx, torch.inverse(Kxx + noise_mat)), (train_y-const_mean)) + const_mean
                var = k - torch.matmul(torch.matmul(Kx, torch.inverse(Kxx + noise_mat)), Kx.T)

                dist = torch.distributions.Normal(mean, var)
                log_prob_j = dist.log_prob(test_y)
                log_prob += log_prob_j.item()

            out_logprob.append(log_prob)

        out_logprob = np.array(out_logprob)
        select_idx = np.argmax(out_logprob)
        self.gp_model = self.gp_models[select_idx]
        self.likelihood = self.likelihoods[select_idx]
    

class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, if_ard=False, if_softplus=True):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        if not if_softplus:
            self.ls_constraint = Positive(transform=torch.exp, inv_transform=torch.log)
        if if_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_constraint=self.ls_constraint),
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(lengthscale_constraint=self.ls_constraint))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_Wrapper:
    def __init__(self, train_x, train_y, if_ard=False, if_softplus=True):
        self.X = train_x
        self.y = train_y.squeeze()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = ExactGPModel(self.X, self.y, self.likelihood, if_ard, if_softplus)

    def train_model(self, epochs=100, lr=0.1):
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.X)
            # Calc loss and backprop gradients
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()

    def pred(self, test_x, num_samples=8):
        self.gp_model.eval()
        f_pred = self.gp_model(test_x)
        means = f_pred.mean
        vars = f_pred.variance
        dist = torch.distributions.MultivariateNormal(
            means.squeeze(),
            torch.diag(vars.squeeze())
        )
        samples = dist.sample((num_samples,)).permute(1, 0)
        return samples, means, vars


class ExactGPModelPyro(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, if_ard=False, if_softplus=True):
        super(ExactGPModelPyro, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        if not if_softplus:
            self.ls_constraint = Positive(transform=torch.exp, inv_transform=torch.log)
        if if_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_constraint=self.ls_constraint),
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(lengthscale_constraint=self.ls_constraint))
        self.fitted = False
        self.num_samples = None

    def _check_if_fitted(self):
        return self.fitted

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_Wrapper_pyro:
    def __init__(self, train_x, train_y, if_ard=False, if_softplus=True):
        self.X = train_x
        self.y = train_y.squeeze()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
        self.gp_model = ExactGPModelPyro(self.X, self.y, self.likelihood, if_ard, if_softplus)
        # this model is used for loading samples from gp_model for easy batched posterior
        self.gp_model_acqf = SaasFullyBayesianSingleTaskGP(self.X, self.y.unsqueeze(-1))

    def train_model(self, warmup_steps=512, num_samples=256, thinning=16):
        self.gp_model.mean_module.register_prior("mean_prior", NormalPrior(0.0, 1.0), "constant")
        self.gp_model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.001, 30.0), "lengthscale")
        # self.gp_model.covar_module.base_kernel.register_prior("lengthscale_prior", GammaPrior(1.5, 0.25),"lengthscale")
        self.gp_model.covar_module.register_prior("outputscale_prior", GammaPrior(2.0, 0.15), "outputscale")
        self.likelihood.register_prior("noise_prior", GammaPrior(0.9, 10.0), "noise")

        def pyro_model(x, y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = self.gp_model.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(x))
                pyro.sample("obs", output, obs=y)
            return y

        nuts_kernel = NUTS(pyro_model)
        mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=False)

        mcmc_run.run(self.X, self.y)
        mcmc_samples = mcmc_run.get_samples()
        for k, v in mcmc_samples.items():
            mcmc_samples[k] = v[::thinning]
        self.gp_model.pyro_load_from_samples(mcmc_samples)
        (
            self.gp_model_acqf.mean_module,
            self.gp_model_acqf.covar_module,
            self.gp_model_acqf.likelihood,
        ) = self.gp_model.mean_module, self.gp_model.covar_module, self.gp_model.likelihood

        self.gp_model.fitted = True
        self.gp_model.num_samples = int(num_samples/thinning)
        self.gp_model.eval()




