from data import *
import pickle
from infras.randutils import *
from benchmark.rover_function import Rover
from benchmark.naslib_benchmark import NasBench201
from benchmark.svm_benchmark import SVMBenchmark
from benchmark.mopta8 import MoptaSoftConstraints
from benchmark.real_dataset import RealDataset
from BO_loop import BO_loop_GP, BO_loop_SaasBO_MAP, BO_loop_GP_pyro
from benchmark.DNA import DNA_Lasso
import click


class Config:
    def __init__(self, func_name, model_name, seed, beta, if_softplus):
        self.func_name = func_name
        self.model_name = model_name
        self.seed = seed
        self.beta = beta
        self.if_softplus = if_softplus


def all_configs():
    config_list = []
    for seed in range(10):
        for model_name in ['GP_ARD', 'GP', 'GP_ARD_PYRO', 'GP_PYRO', 'SaasBO_MAP']:
            for func_name in ['mopta08', 'rover', 'nas201', 'dna', 'SVM', 'Ackley', 'Ackley150', 'StybTang_V1',
                              'Rosenbrock_V1', 'Rosenbrock100_V1', 'Hartmann6']:
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
    beta_out = int(10 * beta)
    if_softplus = config.if_softplus

    print(f"Running --- {func_name}, SEED={SEED}, model={model_name}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_default_tensor_type(torch.DoubleTensor)

    if func_name == 'mopta08':
        num_step = 300
        func = MoptaSoftConstraints()
        dst = RealDataset(func, 20, 'lhs', SEED)
    elif func_name == 'SVM':
        num_step = 300
        func = SVMBenchmark()
        dst = RealDataset(func, 20, 'lhs', SEED)
    elif func_name == 'rover':
        num_step = 300
        func = Rover()
        dst = RealDataset(func, 20, 'lhs', SEED)
    elif func_name == 'nas201':
        num_step = 300
        func = NasBench201()
        dst = RealDataset(func, 20, 'lhs', SEED)
    elif func_name == 'dna':
        num_step = 300
        func = DNA_Lasso()
        dst = RealDataset(func, 20, 'lhs', SEED)

    elif func_name == 'Rosenbrock':
        num_step = 400
        D = 100
        func = FuncRosenbrock(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    elif func_name == 'Ackley':
        num_step = 400
        D = 150
        func = FuncAckley(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    elif func_name == 'StybTang':
        num_step = 400
        D = 200
        func = FuncStybTang(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    elif func_name == 'StybTang_V1':
        num_step = 400
        D = 200
        func = FuncStybTang_V1(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    elif func_name == 'Rosenbrock100':
        num_step = 400
        D = 300
        func = FuncRosenbrock100(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    elif func_name == 'Rosenbrock_V1':
        num_step = 400
        D = 300
        func = FuncRosenbrock_V1(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    elif func_name == 'Rosenbrock100_V1':
        num_step = 400
        D = 300
        func = FuncRosenbrock100_V1(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    elif func_name == 'Ackley150':
        num_step = 400
        D = 300
        func = FuncAckley150(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)
    elif func_name == 'Hartmann6':
        num_step = 400
        D = 300
        func = FuncHartmann6(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    else:
        raise NotImplementedError

    if model_name == 'GP':
        best_val, time_list = BO_loop_GP(dst, SEED, num_step=num_step, beta=beta, if_ard=False, if_softplus=if_softplus)
    elif model_name == 'GP_ARD':
        best_val, time_list = BO_loop_GP(dst, SEED, num_step=num_step, beta=beta, if_ard=True, if_softplus=if_softplus)
    elif model_name == 'GP_ARD_PYRO':
        best_val, time_list = BO_loop_GP_pyro(dst, SEED, num_step=num_step, beta=1.5, if_ard=True, if_softplus=True)
    elif model_name == 'GP_PYRO':
        best_val, time_list = BO_loop_GP_pyro(dst, SEED, num_step=num_step, beta=1.5, if_ard=False, if_softplus=True)
    elif model_name == 'SaasBO_MAP':
        best_val, time_list = BO_loop_SaasBO_MAP(dst, num_step=num_step, acqf="EI")
    else:
        raise NotImplementedError

    BO_result = {
        "time": time_list,
        "X": dst.X,
        "Y": dst.y
    }

    if model_name in ['GP', 'GP_ARD']:
        output_dir = os.path.join(cwd, "gp_output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_{func_name}_{SEED}_{beta_out}_{if_softplus}_10.pickle'")
    elif model_name in ['GP_ARD_PYRO', 'GP_PYRO']:
        output_dir = os.path.join(cwd, "gp_pyro_output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_{func_name}_{SEED}.pickle'")
    elif model_name in ['SaasBO_MAP']:
        output_dir = os.path.join(cwd, "saasbo_output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_{func_name}_{SEED}.pickle")
    else:
        raise NotImplementedError

    with open(output_file, 'wb') as handle:
        pickle.dump(BO_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


