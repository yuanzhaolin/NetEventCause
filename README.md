# NEC

This project is the source code of the paper _NetEventCause: Event-driven Root Cause Analysis for Large Network System without Topology_


## Create the environment

```shell
conda env create -n <myenv> -f config/environment.yml
conda activate <myenv>
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
```

## Starting the training and evaluation

```shell
./scripts/toy_all.sh
```
