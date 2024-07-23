# FCNR: Fast Compressive Neural Representation of Visualization Images
### [paper](*)
Yunfei Lu, Pengfei Gu, Chaoli Wang

This is the official pytorch implementation for the paper "FCNR: Fast Compressive Neural Representation of Visualization Images". 

![image](./figures/overview.png "Overview")

<div style="display: flex; justify-content: space-between;">
  <img src="./figures/architecture1.png" alt="Encoders" style="width: 30%;">
  <img src="./figures/architecture2.png" alt="Decoders" style="width: 30%;">
  <img src="./figures/architecture3.png" alt="Stereo Context Modules" style="width: 30%;">
</div>


## Get Started

Set up a conda environment with all dependencies with Python 3.9:

```
pip install -r requirements.txt
```

## Data 
You can generate customized visualization images with different viewpoints and timesteps on your own dataset via volume or isosurface rendering. Here is a link to download the vortex dataset (direct volume rendering images included) we use: <a href="https://drive.google.com/drive/folders/1oXtXkXnzUrhfcK8YgHmV52jEwd6dbYm0?usp=sharing">vortex</a>. 

## Training & Inference
Specify `<gpu_idx>`, `<exp_name>` and `<config_name>` to start training and inferencing:

```
python train.py <gpu_idx> <exp_name> --config ./configs/<config_name>
```

An example of the configuration file we use is `./configs/cfg.json`. You can follow it to implement on your customized dataset.

## Results
Here is a comparison between the results of FCNR and existing baselines:
<img src="./figures/results.png" alt="Results" height=60% width=60% />

## Citation
```
@article{lu2024fcnr,
  title={{FCNR}: Fast Compressive Neural Representation of Visualization Images},
  author={Lu, Yunfei, Gu, Pengfei, and Wang, Chaoli},
  booktitle={Proceedings of IEEE VIS Conference (Short Papers)},
  year={2024},
  note={Accepted}
}
```




