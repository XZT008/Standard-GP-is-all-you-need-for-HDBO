from data import *
import pickle
from infras.randutils import *
from benchmark.real_dataset import RealDataset
from BO_loop import BO_loop_GP,  BO_loop_GP_MAP, Vanilla_BO_loop
import click
from joblib import Parallel, delayed
from benchmark.MujocoHumanoid import MujocoHumanoid


class Config:
    def __init__(self, func_name, model_name, seed, beta, if_softplus):
        self.func_name = func_name
        self.model_name = model_name
        self.seed = seed
        self.beta = beta
        self.if_softplus = if_softplus


def all_configs():
    config_list = []
    for seed in range(10, 20):
        for model_name in ["GP_ARD", "GP_ARD_setls", "GP_ARD_RBF", "GP_ARD_RBF_setls"]:
            for func_name in ["Humanoid"]:
                for beta in [1.5]:
                    for if_softplus in [True]:
                        config = Config(func_name, model_name, seed, beta, if_softplus)
                        config_list.append(config)
    return config_list


def get_config(index):
    config_l = all_configs()
    print(f"{index} out of {len(config_l)}", flush=True)
    return config_l[index]


@click.command()
@click.option("--index", type=int, required=True, help="Which grid index to run.")
def main(index):
    cwd = os.getcwd()
    config = get_config(index)
    model_name = config.model_name
    SEED = config.seed
    func_name = config.func_name
    beta = config.beta
    if_softplus = config.if_softplus

    print(f"Running --- {func_name}, SEED={SEED}, model={model_name}")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    if func_name == 'mopta08':
        num_step = 300
        func = MujocoHumanoid()
        dst = RealDataset(func, 50, 'lhs', SEED)

    else:
        raise NotImplementedError

    try:
        if model_name == 'GP_ARD':
            best_val, time_list = BO_loop_GP(func_name, dst, SEED, num_step=num_step, beta=beta, if_ard=True,
                                             if_softplus=if_softplus, acqf_type="UCB", if_matern=True, set_ls=False)
        elif model_name == 'GP_ARD_RBF':
            best_val, time_list = BO_loop_GP_MAP(func_name, dst, SEED, num_step=num_step, beta=beta, if_ard=True,
                                                 optim_type="LBFGS", acqf_type="UCB", ls_prior_type="Uniform",
                                                 if_matern=False, set_ls=False)
        elif model_name == "GP_ARD_setls":
            best_val, time_list = BO_loop_GP(func_name, dst, SEED, num_step=num_step, beta=beta, if_ard=True,
                                             if_softplus=if_softplus, acqf_type="UCB", if_matern=True, set_ls=True)
        elif model_name == "GP_ARD_RBF_setls":
            best_val, time_list = BO_loop_GP_MAP(func_name, dst, SEED, num_step=num_step, beta=beta, if_ard=True,
                                                 optim_type="LBFGS", acqf_type="UCB", ls_prior_type="Uniform",
                                                 set_ls=True, if_matern=False)
        elif model_name == "Vanilla_BO":
            best_val, time_list = Vanilla_BO_loop(func_name, dst, SEED, num_step=num_step)
        else:
            raise NotImplementedError

        BO_result = {
            "time": time_list,
            "X": dst.X,
            "Y": dst.y
        }

        output_dir = os.path.join(cwd, model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_{func_name}_{SEED}.pickle")

        with open(output_file, 'wb') as handle:
            pickle.dump(BO_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except:
        print(f"Error in {index}")


if __name__ == "__main__":
    # This is for HPC run
    main()

    # Below is for multiprocess run
    # start = 0
    # end = 10
    # Parallel(n_jobs=(end - start))(delayed(main)(index) for index in range(start, end))