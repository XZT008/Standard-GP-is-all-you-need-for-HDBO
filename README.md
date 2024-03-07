# Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization

This repository provides code for our paper: [Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization](https://arxiv.org/abs/2402.02746).
In our paper, we argued that Standard GP is by far the most robust surrogate model for HDBO.

## Installation

## Running experiments
We wrote our run script `Standard-BO/baselines/run_script.py` in a way that could be efficiently run on HPC with Slurm.
```angular2html
python run_script.py --index=${SLURM_ARRAY_TASK_ID}
```
where index corresponds to the index define in experiment list,
```angular2html
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
```

Here is a sample SBATCH File for running all experiments,
```angular2html
#!/bin/bash
#
#SBATCH --job-name=myjob
#SBATCH --output=./GP_ARD_output/%x_%A_%a.out
#SBATCH --error=./GP_ARD_output/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=10:00:00
#SBATCH --array=0-549

module purge;

singularity exec --overlay /scratch/zx581/BO/overlay-15GB-500K.ext3:ro  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh;
conda activate *YOUR ENV NAME*;
cd *PATH TO run_script.py*;
python run_script.py --index=${SLURM_ARRAY_TASK_ID};
"
```

## BO LOOPS
We provided 3 BO loops for standard GP in `Standard-BO/baselines/BO_loop.py`:
1. BO with standard GP, trained with MLE: ```BO_loop_GP(dataset, seed, num_step=200, beta=1.5, if_ard=False, if_softplus=True, acqf_type="UCB")```. `if_ard` controls if ARD kernel is used. `if_softplus` controls the positive constraint, whether SOFTPLUS or EXP.
2. BO with standard GP, and NUTS sampling: ```BO_loop_GP_pyro(dataset, seed, num_step=200, beta=1.0, if_ard=False, if_softplus=True)```. Our default warmup steps and samples are 512 and 256. One could modify those in `model.train_model(warmup_steps=512, num_samples=256)`.
3. SaasBO with MAP: We implemented [SaasBO with MAP estimator](https://arxiv.org/abs/2103.00349) based on their implementation details. `BO_loop_SaasBO_MAP(dataset, num_step=200, acqf="EI")`.

## Reference
Here are the references for the code of previous real world benchmarks that are used in our experiments:
1. SVM, Mopta08 are taken from [BAxUS](https://github.com/LeoIV/BAxUS/).
2. Rover is taken from [EBO](https://github.com/zi-w/Ensemble-Bayesian-Optimization).
3. NAS201 is taken from [MCTS-VS](https://github.com/lamda-bbo/MCTS-VS).

We really appreciate their contribution!