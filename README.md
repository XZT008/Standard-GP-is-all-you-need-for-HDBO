# Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization

This repository provides code for our paper: [Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization](https://arxiv.org/abs/2402.02746).
In our paper, we argued that Standard GP is by far the most robust surrogate model for HDBO.

## Installation
### Setting environment
NOTE: This code need several repositories for realworld benchmarks, and there will be some dependency conflicts. Please just ignore them.

First, create a conda env and install [LassoBench](https://github.com/ksehic/LassoBench) and [NASLib](https://github.com/automl/NASLib).
```angular2html
conda create -n gp_env python=3.8
conda activate gp_env
pip install --upgrade pip setuptools wheel

# cd TO THIS REPO
git clone https://github.com/ksehic/LassoBench.git
git clone https://github.com/automl/NASLib.git

# First install LassoBench, then NASLib
cd LassoBench
pip install -e .
cd ..

# Please consider change requirements.txt as described in Troubleshooting
cd NASLib 
pip install -e .
cd ..

# Finally, install this repo
pip install -e .
```
## Troubleshooting
If you have trouble installing NASLib, then modify their requirements.txt. Change `numpy>=1.22.0`
to `numpy==1.22.0`.

For Windows users, if installation might fail due to grakel, please change `grakel==0.1.8` to `grakel==0.1.10` in NASLib requirements.txt

### Download data and executables
1. For mopta, download from [Here](https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html). If your machine is amd64, use this [link](https://mopta.papenmeier.io/mopta08_amd64.exe). And put it under `Standard-BO/benchmark/data`.
2. For SVM, download from [Here](https://archive.ics.uci.edu/dataset/206/relative+location+of+ct+slices+on+axial+axis). And put the .csv file under `Standard-BO/benchmark/data`.
3. For NAS201, download nb201_cifar100_full_training.pickle from [Here](https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa). And put it under `/NASLib/naslib/data/`.
## Running experiments
We wrote our run script `Standard-BO/baselines/run_script.py` in a way that could be efficiently run on HPC with Slurm.
```angular2html
python run_script.py --index=${SLURM_ARRAY_TASK_ID}
```
where index corresponds to the index define in experiment list.
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
1. BO with standard GP, trained with MLE: ```BO_loop_GP(dataset, seed, num_step=200, beta=1.5, if_ard=False, if_softplus=True, acqf_type="UCB")```. `if_ard` 
controls if ARD kernel is used. `if_softplus` controls the positive constraint, whether SOFTPLUS or EXP. The default acqf is UCB.
2. BO with standard GP, and NUTS sampling: ```BO_loop_GP_pyro(dataset, seed, num_step=200, beta=1.0, if_ard=False, if_softplus=True)```. Our default warmup steps and samples are 512 and 256. One could modify those in `model.train_model(warmup_steps=512, num_samples=256)`.
3. SaasBO with MAP: We implemented [SaasBO with MAP estimator](https://arxiv.org/abs/2103.00349) based on their implementation details. `BO_loop_SaasBO_MAP(dataset, num_step=200, acqf="EI")`.

## Reference
Here are the references for the code of previous real world benchmarks that are used in our experiments:
1. SVM, Mopta08 are taken from [BAxUS](https://github.com/LeoIV/BAxUS/).
2. Rover is taken from [EBO](https://github.com/zi-w/Ensemble-Bayesian-Optimization).
3. NAS201 is taken from [MCTS-VS](https://github.com/lamda-bbo/MCTS-VS).

We really appreciate their contribution! Especially [BAxUS](https://github.com/LeoIV/BAxUS/) authors, they created the links to download those executables.

## Citation
If you find our work or code useful in your research, you could cite those with following Bibtex:
```
@misc{xu2024standard,
      title={Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization}, 
      author={Zhitong Xu and Shandian Zhe},
      year={2024},
      eprint={2402.02746},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```