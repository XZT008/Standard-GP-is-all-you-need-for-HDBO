import numpy as np
import torch
import gpytorch
import math
from data import *
import random
from tqdm import tqdm
#import ssl
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.gpytorch import GPyTorchModel
from torch.quasirandom import SobolEngine
from gpytorch.constraints import Positive, Interval
from gpytorch.priors import HalfCauchyPrior, LogNormalPrior, GammaPrior, UniformPrior, MultivariateNormalPrior, NormalPrior
from gpytorch.functions import MaternCovariance

from gpytorch.kernels.kernel import Kernel
from typing import Optional

from infras.randutils import *

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from pyro.infer.mcmc import NUTS, MCMC
import pyro
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
import botorch
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from botorch.optim.stopping import ExpMAStoppingCriterion

#torch.manual_seed(0)
#np.random.seed(0)
#random.seed(0)


def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)


class Vanilla_GP_Wrapper:
    def __init__(self, train_x, train_y):
        self.X = train_x
        self.dim = self.X.shape[1]
        self.y = train_y.reshape(-1, 1)

        ls_prior = LogNormalPrior(math.sqrt(2)+(math.log(self.dim)/2.0), math.sqrt(3))
        covar_module = ScaleKernel(
            base_kernel=RBFKernel(
                ard_num_dims=train_x.shape[1],
                nu=2.5,
                lengthscale_prior=ls_prior,
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        self.gp_model = SingleTaskGP(self.X, self.y, covar_module=covar_module)

    def train_model(self):
        self.gp_model.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        optimizer = torch.optim.RMSprop(mll.parameters(), lr=0.1)
        botorch.fit.fit_gpytorch_mll_torch(mll, optimizer=optimizer)

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


class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, if_ard=True, if_softplus=True, set_ls=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        self.D = train_x.shape[1]

        if if_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_constraint=self.ls_constraint),
            )
            if set_ls:
                ls = torch.ones_like(self.covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
                self.covar_module.base_kernel._set_lengthscale(ls)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(lengthscale_constraint=self.ls_constraint))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelRBF(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, if_ard=True, if_softplus=True, set_ls=False):
        super(ExactGPModelRBF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        self.D = train_x.shape[1]
        if not if_softplus:
            self.ls_constraint = Positive(transform=torch.exp, inv_transform=torch.log)
        if if_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1], lengthscale_constraint=None),
            )
            if set_ls:
                ls = torch.ones_like(self.covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
                self.covar_module.base_kernel._set_lengthscale(ls)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=self.ls_constraint))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_Wrapper:
    def __init__(self, train_x, train_y, if_ard=False, if_softplus=True, if_matern=True, set_ls=False, device="cpu"):
        self.device = device
        self.X = train_x
        self.y = train_y.squeeze()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        if if_matern:
            self.gp_model = ExactGPModel(self.X, self.y, self.likelihood, if_ard, if_softplus, set_ls=set_ls).to(self.device)
        else:
            self.gp_model = ExactGPModelRBF(self.X, self.y, self.likelihood, if_ard, if_softplus, set_ls=set_ls).to(self.device)

    def train_model(self, epochs=500, lr=0.1, optim="ADAM"):
        self.gp_model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model).to(self.device)
        if optim == "ADAM":
            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        elif optim == "RMSPROP":
            optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=lr)
        elif optim == "botorch":
            stop_c = ExpMAStoppingCriterion(
                maxiter=10000,
                minimize=True,
                n_window=10,
                eta=1.0,
                rel_tol=1e-6,
            )

            optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=0.05)
            botorch.fit.fit_gpytorch_mll_torch(mll, optimizer=optimizer, step_limit=1500, stopping_criterion=stop_c)
            return
        else:
            raise NotImplementedError

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.X)
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


class GP_MAP_Wrapper:
    def __init__(self, train_x, train_y, if_ard=True, ls_prior_type="Uniform", optim_type="LBFGS", set_ls=False,
                 if_matern=True, device="cpu"):
        self.device = device
        self.X = train_x
        self.D = train_x.shape[1]
        self.y = train_y.reshape(-1, 1)
        self.optim_type = optim_type
        if ls_prior_type == "Gamma":
            ls_prior = GammaPrior(3.0, 6.0)
            ls_constraint = None
        elif ls_prior_type == "Uniform":
            if self.D >= 100:
                ls_prior = UniformPrior(1e-10, 30)
                ls_constraint = Interval(lower_bound=1e-10, upper_bound=30)
            else:
                ls_prior = UniformPrior(1e-10, 10.0)
                ls_constraint = Interval(lower_bound=1e-10, upper_bound=10.0)
        else:
            raise NotImplementedError

        if if_ard and if_matern:
            covar_module = ScaleKernel(
                base_kernel=MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_prior=ls_prior,
                                                    lengthscale_constraint=ls_constraint),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            if set_ls:
                ls = torch.ones_like(covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
                covar_module.base_kernel._set_lengthscale(ls)
            else:
                ls = torch.ones_like(covar_module.base_kernel.lengthscale) * 0.6931
                covar_module.base_kernel._set_lengthscale(ls)

        elif if_ard and not if_matern:
            covar_module = ScaleKernel(
                base_kernel=RBFKernel(ard_num_dims=train_x.shape[1], lengthscale_prior=ls_prior,
                                         lengthscale_constraint=ls_constraint),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            if set_ls:
                ls = torch.ones_like(covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
                covar_module.base_kernel._set_lengthscale(ls)
            else:
                ls = torch.ones_like(covar_module.base_kernel.lengthscale) * 0.6931
                covar_module.base_kernel._set_lengthscale(ls)

        else:
            raise NotImplementedError

        self.gp_model = SingleTaskGP(self.X, self.y, covar_module=covar_module).to(self.device)

    def train_model(self):
        self.gp_model.train()
        if self.optim_type == "LBFGS":
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model).to(self.device)
            botorch.fit.fit_gpytorch_mll(mll)
        elif self.optim_type == "ADAM":
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model).to(self.device)
            stop_c = ExpMAStoppingCriterion(
                maxiter = 10000,
                minimize = True,
                n_window = 10,
                eta = 1.0,
                rel_tol = 1e-6,
                )

            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.05)
            botorch.fit.fit_gpytorch_mll_torch(mll, optimizer=optimizer, step_limit=1000, stopping_criterion=stop_c)

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
