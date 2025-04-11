import torch
from gp_ablation import GP_Wrapper
from joblib import Parallel, delayed
from data import *
import pickle
from infras.randutils import *
import math


class Config:
    def __init__(self, func_name, seed, init_ls, kernel_type, dim):
        self.func_name = func_name
        self.seed = seed
        self.init_ls = init_ls
        self.kernel_type = kernel_type
        self.dim = dim


def all_config():
    config_list = []
    for seed in range(20):
        for dim in [50, 100, 200, 300, 400, 500, 600]:
            for func_name in ["Hartmann6", "Rosenbrock_V1"]:
                for kernel_type in ["matern", "rbf"]:
                    for init_ls in [0.1, 0.5, 0.6931, 1.0, 3.0, "sqrt_d"]:
                        config = Config(func_name, seed, init_ls, kernel_type, dim)
                        config_list.append(config)

    return config_list


def run(train_x, train_y, test_x, test_y, init_ls, kernel_type):
    if kernel_type == "matern":
        if_matern = True
    elif kernel_type == "rbf":
        if_matern = False
    else:
        raise NotImplementedError

    gp_model = GP_Wrapper(train_x, train_y, if_matern=if_matern, set_ls=init_ls, gamma_prior=False)
    ls, raw_ls_grad, cov_matrix = gp_model.train_model_ADAM(epochs=1500, lr=0.1)

    ls_start = ls[0]
    ls_end = ls[-1]
    ls_diff = ls_end - ls_start
    rel_l2_ls = np.linalg.norm(ls_diff, ord=2) / np.linalg.norm(ls_start, ord=2)
    raw_ls_grad_norm = np.linalg.norm(raw_ls_grad, axis=1)
    max_ratio_l = []
    for cov_mat in cov_matrix:
        diag = np.diag(cov_mat)
        cov_mat_abs = np.sum(np.abs(cov_mat), axis=1)

        ratio = cov_mat_abs / diag
        max_ratio_l.append(max(ratio))

    # testing for mse and log likelihood
    _, test_y_pred_mean, test_y_pred_var = gp_model.pred(test_x)
    mse = ((test_y.reshape(-1) - test_y_pred_mean.reshape(-1)) ** 2).mean()

    dist = torch.distributions.Normal(loc=test_y_pred_mean, scale=torch.sqrt(test_y_pred_var))
    ll = dist.log_prob(test_y.reshape(-1)).mean()
    data = {
        "rel l2 diff ls": rel_l2_ls,
        "raw ls grad norm": raw_ls_grad_norm,
        "max ratio": max_ratio_l,
        "ll": ll.item(),
        "mse": mse.item()
    }

    return data


def main(index):
    cwd = os.getcwd()
    config_l = all_config()

    if index >= len(config_l):
        print("All experiments are done")
        return 0

    print(f"Running {index} out of all {len(config_l)} experiments")
    config = config_l[index]

    # get parameters
    SEED = config.seed
    func_name = config.func_name
    init_ls = config.init_ls
    kernel_type = config.kernel_type
    dim = config.dim

    print(f"Running --- {func_name}, SEED:{SEED}, LS:{init_ls}, kernel={kernel_type}, dim={dim}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_default_tensor_type(torch.DoubleTensor)

    N_train = 500
    N_test = 100
    N = N_train + N_test

    if func_name == 'Hartmann6':
        D = dim
        func = FuncHartmann6(D, maximize=True)
        dst = BayesOptDataset(func, N, 'lhs', SEED)
    elif func_name == 'Rosenbrock_V1':
        D = dim
        func = FuncRosenbrock_V1(D, maximize=True)
        dst = BayesOptDataset(func, N, 'lhs', SEED)
    else:
        raise NotImplementedError

    if init_ls == "sqrt_d":
        init_ls = math.sqrt(func.dims)
    x, y = dst.get_data(normalize=True)

    train_x, test_x = x[:N_train], x[N_train:]
    train_y, test_y = y[:N_train], y[N_train:]

    ret_data = run(train_x, train_y, test_x, test_y, init_ls, kernel_type)

    file_name = f"{func_name}_{SEED}_{init_ls}_{kernel_type}_{dim}.pickle"
    output_dir = os.path.join(cwd, "ablation_output_dim", file_name)
    with open(output_dir, 'wb') as handle:
        pickle.dump(ret_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    for i in range(34):
        start = i * 100
        end = start + 100
        Parallel(n_jobs=(end - start))(delayed(main)(index) for index in range(start, end))


























