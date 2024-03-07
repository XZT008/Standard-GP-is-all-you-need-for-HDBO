import numpy as np
import torch
import gpytorch
import math

from botorch import fit_fully_bayesian_model_nuts
from botorch.models import SaasFullyBayesianSingleTaskGP

import random
from tqdm import tqdm
import ssl
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.gpytorch import GPyTorchModel
from torch.quasirandom import SobolEngine
import pickle
from infras.randutils import *
from benchmark.rover_function import Rover
from benchmark.naslib_benchmark import NasBench201
from benchmark.svm_benchmark import SVMBenchmark
from benchmark.mopta8 import MoptaSoftConstraints
from benchmark.real_dataset import RealDataset
from baselines.GP import GP_Wrapper, SaasGP_MAP_Wrapper, GP_Wrapper_pyro
from benchmark.DNA import DNA_Lasso
from data import *
import time
from infras.randutils import *


def BO_loop_GP(dataset, seed, num_step=200, beta=1.5, if_ard=False, if_softplus=True, acqf_type="UCB"):
    best_y = []
    time_list = []
    dim = dataset.func.dims

    for i in range(1, num_step+1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        best_y_before = dataset.get_curr_max_unnormed()
        model = GP_Wrapper(X, Y, if_ard, if_softplus)
        model.train_model(500, 0.1)

        if acqf_type == "UCB":
            acqf = UpperConfidenceBound(model=model.gp_model, beta=beta, maximize=True)
        elif acqf_type == "EI":
            acqf = ExpectedImprovement(model=model.gp_model, best_f=Y.max())
        else:
            raise NotImplementedError

        new_x, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
            q=1,
            num_restarts=10,
            raw_samples=1000,
            options={},
        )
        
        end_time = time.time()
        time_used = end_time - start_time
        time_list.append(time_used)
        dataset.add(new_x)
        best_y_after = dataset.get_curr_max_unnormed()
        
        print(f"Seed: {seed} --- At itr: {i}: best value before={best_y_before}, best value after={best_y_after}, current query: {dataset.y[-1]}", flush=True)
        best_y.append(best_y_before)
    return best_y, time_list
    

def BO_loop_SaasBO(dataset, num_step=200, acqf="EI"):
    best_y = []
    time_list = []
    dim = dataset.func.dims
    for i in range(1, num_step+1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        best_y_before = dataset.get_curr_max_unnormed()
        gp = SaasFullyBayesianSingleTaskGP(
            train_X=X.to(torch.float64),
            train_Y=Y.to(torch.float64),
            train_Yvar=torch.full_like(Y, 1e-6).to(torch.float64),
        )
        
        fit_fully_bayesian_model_nuts(
            gp,
            warmup_steps=512,
            num_samples=256,
            thinning=16,
            disable_progbar=True,
        )
        
        if acqf == "EI":
            acqf = ExpectedImprovement(model=gp, best_f=Y.max().to(torch.float64))
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
                q=1,
                num_restarts=10,
                raw_samples=1024,
                options={},
            )
        else:
            acqf = UpperConfidenceBound(model=gp, beta=1.5, maximize=True)
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
                q=1,
                num_restarts=10,
                raw_samples=1000,
                options={},
            )
        end_time = time.time()
        time_used = end_time - start_time
        time_list.append(time_used)
        dataset.add(new_x)
        best_y_after = dataset.get_curr_max_unnormed()
        print(f"At itr: {i}: best value before={best_y_before}, best value after={best_y_after}", flush=True)
        print(f"Time used in {i}: {end_time - start_time}", flush=True)
        best_y.append(best_y_before)
        del gp
    return best_y, time_list


def BO_loop_SaasBO_MAP(dataset, num_step=200, acqf="EI"):
    best_y = []
    time_list = []
    dim = dataset.func.dims
    for i in range(1, num_step+1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        best_y_before = dataset.get_curr_max_unnormed()
        model = SaasGP_MAP_Wrapper(X.to(torch.float64), Y.to(torch.float64))
        model.train_model(1500, 0.02)
        if acqf == "EI":
            acqf = ExpectedImprovement(model=model.gp_model, best_f=Y.max().to(torch.float64))
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
                q=1,
                num_restarts=5,
                raw_samples=5000,
                options={},
            )
        else:
            acqf = UpperConfidenceBound(model=model.gp_model, beta=1.5, maximize=True)
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
                q=1,
                num_restarts=5,
                raw_samples=5000,
                options={},
            )
        end_time = time.time()
        time_used = end_time - start_time
        time_list.append(time_used)
        dataset.add(new_x)
        best_y_after = dataset.get_curr_max_unnormed()
        print(f"At itr: {i}: best value before={best_y_before}, best value after={best_y_after}", flush=True)
        best_y.append(best_y_before)
    return best_y, time_list


def BO_loop_GP_pyro(dataset, seed, num_step=200, beta=1.0, if_ard=False, if_softplus=True):
    best_y = []
    time_list = []
    dim = dataset.func.dims
    for i in range(1, num_step+1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        best_y_before = dataset.get_curr_max_unnormed()
        model = GP_Wrapper_pyro(X, Y, if_ard, if_softplus)
        model.train_model(warmup_steps=512, num_samples=256)
        model.gp_model.eval()

        acqf = UpperConfidenceBound(model=model.gp_model_acqf, beta=beta)
        # acqf = ExpectedImprovement(model=model.gp_model_acqf, best_f=Y.max())
        new_x, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
            q=1,
            num_restarts=5,
            raw_samples=1000,
            options={},
        )
        end_time = time.time()
        time_used = end_time - start_time
        time_list.append(time_used)
        dataset.add(new_x)
        best_y_after = dataset.get_curr_max_unnormed()
        print(f"SEED: {seed} --- At itr: {i}: best value before={best_y_before}, best value after={best_y_after}, current query: {dataset.y[-1]}", flush=True)
        best_y.append(best_y_before)
    return best_y, time_list

    
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    func = MoptaSoftConstraints()
    dst = RealDataset(func, 20, 'lhs', 0)

    best_val, time_list = BO_loop_SaasBO(dst, num_step=320)
    BO_result = {
        "time": time_list,
        "X": dst.X,
        "Y": dst.y
    }
    with open('test.pickle', 'wb') as handle:
        pickle.dump(BO_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


