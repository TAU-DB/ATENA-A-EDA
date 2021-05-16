# ATENA
This repository contains basic implementation of ATENA, a system for auto-generating Exploratory Data Analysis (EDA) sessions <a href="https://dl.acm.org/doi/abs/10.1145/3318464.3389779">  [Milo et al., SIGMOD '20] </a>. The repository is free for use for academic purposes. Upon using, please cite the paper:</br>
```Ori Bar El, Tova Milo, and Amit Somech. 2020. Automatically Generating Data Exploration Sessions Using Deep Reinforcement Learning. SIGMOD ’20, 1527–1537. DOI:https://doi.org/10.1145/3318464.3389779```

## Setup
Clone the repository and install requirements using the requirements.txt file.
</br> 
ATENA was tested on Conda environment, using Python 3.7.3

## Getting Started
Example command line for training ATENA:
```
$ python train.py --env ATENAcont-v0 --schema FLIGHTS --dataset-number 1 --algo chainerrl_ppo --arch FFParamSoftmax --episode-length 10 --steps 100000 --eval-interval 10000 --stack-obs-num 3 --num-envs 64
```
Run `train.py --help` for further options and documentation.

After training, you will get an output directory similar to this: `results/20210101T123456.718192/10000_finish`
For testing your model, run `test.py` with the mentioned path and the same parameters from training.
In our example, the command line should be:
```
$ python test.py --load results/20210101T123456.718192/10000_finish --env ATENAcont-v0 --schema FLIGHTS --dataset-number 1 --algo chainerrl_ppo --arch FFParamSoftmax --episode-length 10 --stack-obs-num 3
``` 
The output will be an auto-generated EDA session.

 


