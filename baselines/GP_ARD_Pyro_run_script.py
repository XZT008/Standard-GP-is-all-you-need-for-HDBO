from data import *
import pickle
from infras.randutils import *
from benchmark.rover_function import Rover
from benchmark.naslib_benchmark import NasBench201
from benchmark.svm_benchmark import SVMBenchmark
from benchmark.mopta8 import MoptaSoftConstraints
from benchmark.real_dataset import RealDataset
from BO_loop import BO_loop_GP_pyro
from benchmark.DNA import DNA_Lasso
import click
from joblib import Parallel, delayed


def run_parallel(SEED, func_name):
    print(f"Running {func_name} --- seed={SEED}", flush=True)
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
        
    elif func_name == 'Rosenbrock_V1':
        num_step = 400
        D = 100
        func = FuncRosenbrock_V1(D, maximize=True)
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
    
    best_val, time_list = BO_loop_GP_pyro(dst, SEED, num_step=num_step, beta=1.5, if_ard=True, if_softplus=True)
    BO_result = {
        "time": time_list,
        "X": dst.X,
        "Y": dst.y
    }
    
    with open(f'/scratch/zx581/bo_baseline/baselines/gp_ard_pyro/GP_ARD_PYRO_{func_name}_{SEED}_EI.pickle', 'wb') as handle:
        pickle.dump(BO_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

@click.command()
@click.option("--index", type=int, required=True, help="Which grid index to run.")
def main(index):
    print("START!", flush=True)
    func_names = ['mopta08', 'rover', 'nas201', 'dna', 'SVM', 'Ackley', 'Ackley150', 'StybTang_V1', 'Rosenbrock_V1', 'Rosenbrock100_V1', 'Hartmann6']
    assert index <= len(func_names)
    func_name_ = func_names[index]
    seeds = list(range(10))
    Parallel(n_jobs=len(seeds))(delayed(run_parallel)(seed, func_name_) for seed in seeds)


if __name__ == "__main__":
    main()


