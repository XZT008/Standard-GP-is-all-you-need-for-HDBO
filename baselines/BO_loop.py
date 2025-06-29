from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound, LogExpectedImprovement
from botorch.optim import optimize_acqf
from baselines.GP import GP_Wrapper, GP_MAP_Wrapper, Vanilla_GP_Wrapper
from data import *
import time
from infras.randutils import *


def BO_loop_GP(func_name, dataset, seed, num_step=200, beta=1.5, if_ard=False, if_softplus=True, acqf_type="UCB", if_matern=True, set_ls=False,

               device="cpu"):
    best_y = []
    time_list = []
    dim = dataset.func.dims

    for i in range(1, num_step+1):
        start_time = time.time()

        X, Y = dataset.get_data(normalize=True)
        X, Y = X.to(device), Y.to(device)
        best_y_before = dataset.get_curr_max_unnormed()
        model = GP_Wrapper(X, Y, if_ard, if_softplus, if_matern=if_matern, set_ls=set_ls)

        if func_name in ["Ackley150"]:
            model.train_model(1000, 0.01)
        elif func_name in ["Ackley"]:
            # For stability across different cpu platform
            model.train_model(None, None, optim="botorch")
        elif func_name == "Hartmann6":
            # We used RMSProp here, due to adam being not efficient
            model.train_model(400, 0.01, optim="RMSPROP")
        else:
            model.train_model(500, 0.1)

        if acqf_type == "UCB":
            acqf = UpperConfidenceBound(model=model.gp_model, beta=beta, maximize=True).to(device)
        elif acqf_type == "EI":
            acqf = ExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
        elif acqf_type == "LogEI":
            acqf = LogExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
        else:
            raise NotImplementedError

        try:
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
                q=1,
                num_restarts=10,
                raw_samples=1000,
                options={},
            )

        except:
            print(f"ERROR during opt acqf, using random point")
            new_x = torch.rand(dim)
        
        end_time = time.time()
        time_used = end_time - start_time
        time_list.append(time_used)
        dataset.add(new_x)
        best_y_after = dataset.get_curr_max_unnormed()

        itr = dataset.X.shape[0]
        print(f"Seed: {seed} --- At itr: {itr}: best value before={best_y_before}, best value after={best_y_after}, current query: {dataset.y[-1]}", flush=True)
        best_y.append(best_y_before)
    return best_y, time_list


def Vanilla_BO_loop(func_name, dataset, seed, num_step=200):
    """
    Our implementation for Vanilla BO
    """
    best_y = []
    time_list = []
    dim = dataset.func.dims

    for i in range(1, num_step + 1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        best_y_before = dataset.get_curr_max_unnormed()
        model = Vanilla_GP_Wrapper(X, Y)
        model.train_model()

        ls = model.gp_model.covar_module.base_kernel.lengthscale
        print(f"ls mean: {ls.mean()}, ls std: {ls.std()}, max: {ls.max()}, min: {ls.min()}")

        acqf = LogExpectedImprovement(model=model.gp_model, best_f=Y.max())
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

        print(
            f"Seed: {seed} --- At itr: {i}: best value before={best_y_before}, best value after={best_y_after}, current query: {dataset.y[-1]}",
            flush=True)
        best_y.append(best_y_before)
    return best_y, time_list


def BO_loop_GP_MAP(func_name, dataset, seed, num_step=200, beta=1.5, if_ard=True, optim_type="LBFGS", acqf_type="UCB",
                   ls_prior_type="Gamma", set_ls=False, if_matern=False, device="cpu"):
    best_y = []
    time_list = []
    dim = dataset.func.dims
    for i in range(1, num_step + 1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        X = X.to(device)
        Y = Y.to(device)
        best_y_before = dataset.get_curr_max_unnormed()
        model = GP_MAP_Wrapper(X, Y, if_ard=if_ard, if_matern=if_matern, optim_type=optim_type,
                               ls_prior_type=ls_prior_type, device=device, set_ls=set_ls)
        model.train_model()

        if acqf_type == "UCB":
            acqf = UpperConfidenceBound(model=model.gp_model, beta=beta, maximize=True).to(device)
        elif acqf_type == "EI":
            acqf = ExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
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
        print(
            f"Seed: {seed} --- At itr: {i}: best value before={best_y_before}, best value after={best_y_after}, current query: {dataset.y[-1]}",
            flush=True)
        best_y.append(best_y_before)
    return best_y, time_list


