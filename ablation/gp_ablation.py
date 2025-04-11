import numpy as np
import torch
import gpytorch
import math
from data import *
import random
from tqdm import tqdm
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import Positive, Interval
from gpytorch.priors import GammaPrior
import pickle
from infras.randutils import *
from botorch.optim.fit import fit_gpytorch_mll_scipy

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, if_ard=True, if_softplus=True, set_ls=None, gamma_prior=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        ls_prior = GammaPrior(3.0, 6.0)
        if not if_softplus:
            self.ls_constraint = Positive(transform=torch.exp, inv_transform=torch.log)
        if if_ard:
            if gamma_prior:
                self.covar_module = gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1],
                                                                  lengthscale_constraint=self.ls_constraint,
                                                                  lengthscale_prior=ls_prior
                                                                  )
            else:
                self.covar_module = gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1],
                                                                  lengthscale_constraint=self.ls_constraint,
                                                                  )
            if set_ls is not None:
                ls = torch.ones_like(self.covar_module.lengthscale) * set_ls
                self.covar_module._set_lengthscale(ls)
                print()
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(lengthscale_constraint=self.ls_constraint))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelRBF(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, if_ard=True, if_softplus=True, set_ls=None, gamma_prior=False):
        super(ExactGPModelRBF, self).__init__(train_x, train_y, likelihood)
        self.D = train_x.shape[1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        ls_prior = GammaPrior(3.0, 6.0)
        if not if_softplus:
            self.ls_constraint = Positive(transform=torch.exp, inv_transform=torch.log)
        if if_ard:
            if gamma_prior:
                self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1],
                                                               lengthscale_constraint=self.ls_constraint,
                                                               lengthscale_prior=ls_prior)
            else:
                self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1],
                                                               lengthscale_constraint=self.ls_constraint,
                                                                )
            if set_ls is not None:
                ls = torch.ones_like(self.covar_module.lengthscale) * set_ls
                self.covar_module._set_lengthscale(ls)

        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(lengthscale_constraint=self.ls_constraint))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_Wrapper:
    def __init__(self, train_x, train_y, if_ard=True, if_softplus=True, if_matern=True, set_ls=None,
                 gamma_prior=False):
        self.X = train_x
        self.D = train_x.shape[1]
        self.y = train_y.squeeze()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.if_matern = if_matern
        if if_matern:
            self.gp_model = ExactGPModel(self.X, self.y, self.likelihood, if_ard, if_softplus, set_ls=set_ls,
                                         gamma_prior=gamma_prior)
        else:
            self.gp_model = ExactGPModelRBF(self.X, self.y, self.likelihood, if_ard, if_softplus, set_ls=set_ls,
                                            gamma_prior=gamma_prior)

    def train_model_ADAM(self, epochs=1500, lr=0.1):
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        ls_l = []
        raw_ls_grad_l = []
        cov_matrix_l = []
        for _ in tqdm(range(epochs)):
            optimizer.zero_grad()
            output = self.gp_model(self.X)
            # Calc loss and backprop gradients
            loss = -mll(output, self.y)
            loss.backward()
            grad_val = self.gp_model.covar_module.raw_lengthscale.grad.detach().numpy()
            optimizer.step()

            # post-ops for ablation
            ls = self.gp_model.covar_module.lengthscale.detach().numpy()
            cov_matrix = self.gp_model.covar_module(self.X).float().numpy()

            ls_l.append(ls)
            raw_ls_grad_l.append(grad_val)
            cov_matrix_l.append(cov_matrix)

        ls_l = np.concatenate(ls_l, axis=0)
        raw_ls_grad_l = np.concatenate(raw_ls_grad_l, axis=0)
        cov_matrix_l = np.stack(cov_matrix_l, axis=0)

        return ls_l, raw_ls_grad_l, cov_matrix_l

    def train_model_LBFGS(self):
        softplus = torch.nn.Softplus()
        ls_l = []
        raw_ls_grad_l = []
        cov_matrix_l = []

        def store_iteration(xk, _):
            ls_l.append(
                softplus(xk['model.covar_module.raw_lengthscale'].detach().clone()).numpy()
            )
            raw_ls_grad_l.append(xk['model.covar_module.raw_lengthscale'].grad.detach().clone().numpy())

        self.gp_model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        fit_gpytorch_mll_scipy(mll, callback=store_iteration)

        if self.if_matern:
            tmp_covar_module = gpytorch.kernels.MaternKernel(ard_num_dims=self.X.shape[1])
        else:
            tmp_covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=self.X.shape[1])

        for ls in ls_l:
            tmp_covar_module._set_lengthscale(ls)
            cov_matrix = tmp_covar_module(self.X).float().numpy()
            cov_matrix_l.append(cov_matrix)

        ls_l = np.concatenate(ls_l, axis=0)
        raw_ls_grad_l = np.concatenate(raw_ls_grad_l, axis=0)
        cov_matrix_l = np.stack(cov_matrix_l, axis=0)
        return ls_l, raw_ls_grad_l, cov_matrix_l

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


if __name__ == "__main__":
    N_train = 500
    N_test = 100
    N = N_train + N_test
    dim = 300

    func = FuncHartmann6(dim, maximize=True)
    dst = BayesOptDataset(func, N, 'lhs', 0)
    x, y = dst.get_data(normalize=True)
    train_x, test_x = x[:N_train], x[N_train:]
    train_y, test_y = y[:N_train], y[N_train:]
    ls_init_l = [0.1, 0.5, 0.6931, 1.0, 10, math.sqrt(dim)]

    mse_l = []
    for ls_init in ls_init_l:
        gp_model = GP_Wrapper(train_x, train_y, if_matern=True, set_ls=ls_init, gamma_prior=False)

        ls, raw_ls_grad, cov_matrix = gp_model.train_model_LBFGS()
        ls_norm = np.linalg.norm(ls, axis=1)
        raw_ls_grad_norm = np.linalg.norm(raw_ls_grad, axis=1)

        max_ratio_l = []
        for cov_mat in cov_matrix:
            diag = np.diag(cov_mat)
            cov_mat_abs = np.sum(np.abs(cov_mat), axis=1)

            ratio = cov_mat_abs / diag
            max_ratio_l.append(max(ratio))

        data = {
            "ls norm": ls_norm,
            "raw ls grad norm": raw_ls_grad_norm,
            "max ratio": max_ratio_l
        }
        with open(f'./ls_ablation/matern_{ls_init}_h6_lbfgs_g36.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        """
        gp_model.train_model_LBFGS(epochs=1500)
        _, y_pred, _ = gp_model.pred(test_x)
        mse_err = ((y_pred - test_y.squeeze(-1))**2).mean()
        mse_l.append(mse_err.item())
        """
    print()
