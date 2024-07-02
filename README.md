# FCNR: Fast Compressive Neural Representation of Visualization Images

Yunfei Lu, Pengfei Gu, Chaoli Wang

This is the official pytorch implementation for our _IEEE VIS 2024_ short paper "FCNR: Fast Compressive Neural Representation of Visualization Images". 

### Method Overview

![image](./figures/overview.png")

### Get Started

Set up a conda environment with all dependencies with Python 3.9:

```
pip install -r requirements.txt
```

### Training FCNR
Specify `<gpu_idx>`, `<exp_name>` and `<config_name>` to start training:

```
python train.py <gpu_idx> <exp_name> --config ./configs/<config_name>
```

An example of configuration file we use is `./configs/cfg.json`.


