# Bayesian Optimisation for Constrained Problems

## Usage Instructions

To access the data used to generate the plots in the "Bayesian Optimisation for Constrained Problems" paper access
the folder ./cKG/paper_data

### STEP 1: Installation

1. Create virtual environment with the required packages. If you want to create a virtual environment through pip
or anaconda, use the following commands repectively. We use python 3.9.

pip:
```
python3 -m venv env
source env/bin/activate
pip install -r pip_requirements.txt
```

conda:
```
conda create --name <env> --file conda_requirements.txt
```

In case of manual installation, consider installing: matplotlib=3.4.2, numpy=1.20.2, pandas=1.2.4, paramz=0.9.5, pydoe=0.3.8,
scikit-learn=0.24.2, scipy=1.6.3, tensorflow=2.4.1, pandas=1.2.4. 

### STEP 2: Running experiments

1. Access to the experiment folder in ./cKG/core/acquisition. Experiment files are structured as, 
<function_name>_experiment_<method>_<deterministic/noisy>.py. 
To run a file use in the terminal: ` python <function_name>_experiment_<method>_<deterministic/noisy>.py --repeat <integer>`. 
The `--repeat <integer>` parameter sets the random seed in the running file. Outputs will be saved in /cKG/core/acquisition/RESULTS.



