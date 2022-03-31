# Semi-Markov Afterstate Actor-Critic (SMAAC)
This repository is the official implementation of [Winning the L2RPN Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic](https://openreview.net/forum?id=LmUJqB1Cz8).

## Environment setting
ubuntu /linux
下载anaconda
https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh


### Create conda environment
```sh
conda env create -f install.yml
conda activate smaac
```

### lightsim2grid installation
```sh

git clone https://github.com/scalaboy/lightsim2grid
cd lightsim2grid
git submodule init
git submodule update
make clean
make
pip install -U pybind11
pip install -U .
```

## Data download
Since chronic data is required to train or evaluate, please [Download](https://drive.google.com/file/d/15oW1Wq7d6cu6EFS2P7A0cRhyv8u_UqWA/view?usp=sharing).  
Then, replace `data/` with it.
```sh
cd SMAAC
rm -rf data
tar -zxvf data.tar.gz
```

## Scripts
### Train
The detail of arguments is provided in `test.py`.
```sh
python test.py -n=[experiment_name] -s=[seed] -c=[environment_name (5, sand, wcci)]

# Example
python test.py -n=5_run -s=0 -c=5

get result
[  98] Valid: score 97.79145444923583 | step 864.0
[Test Ch  17( 0)] 864/864 ( 96) Score:   98.7509
[Test Ch  17( 1)] 864/864 (  8) Score:   98.5088
[Test Ch  17( 2)] 864/864 (122) Score:   98.3613
[Test Ch  17( 3)] 864/864 (100) Score:   97.5027
[Test Ch  17( 4)] 864/864 (  8) Score:   95.8336
[  99] Valid: score 97.79145444923583 | step 864.0
[Test Ch  17( 0)] 864/864 ( 96) Score:   98.7509
[Test Ch  17( 1)] 864/864 (  8) Score:   98.5088
[Test Ch  17( 2)] 864/864 (122) Score:   98.3613
[Test Ch  17( 3)] 864/864 (100) Score:   97.5027
[Test Ch  17( 4)] 864/864 (  8) Score:   95.8336
[ 100] Valid: score 97.79145444923583 | step 864.0

or
python test.py -n=wcci_run -s=0 -c=wcci
```

### Evaluate
The detail of arguments is provided in `evaluate.py`.
```sh
python evaluate.py -n=[experiment_dirname] -c=[environment_name]

# Example
python evaluate.py -n=wcci_run_0 -c=wcci

# If you want to evaluate an example trained model on WCCI, execute as below
python evaluate.py -n=example
```

## References
```bibtex
@inproceedings{yoon2021winning,
    title={Winning the L2{\{}RPN{\}} Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic},
    author={Deunsol Yoon and Sunghoon Hong and Byung-Jun Lee and Kee-Eung Kim},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=LmUJqB1Cz8}
}
```

## Credit
Our code is based on rte-france's Grid2Op (https://github.com/rte-france/Grid2Op)

# License Information
Copyright (c) 2020 KAIST-AILab

This source code is subject to the terms of the Mozilla Public License (MPL) v2 also available [here](https://www.mozilla.org/en-US/MPL/2.0/)
