# Unofficial Implementation of Sort-Free Gaussian Splatting

This is an unofficial implementation of the ICLR'25 paper "Sort-free Gaussian Splatting via Weighted Sum Rendering". [Paper](https://arxiv.org/pdf/2410.18931)

Our implementation builds upon the original 3DGS by incorporating global parameters `sigma` and `weight_background`, as well as a set of `sh` parameters for each GS to decode the view-dependent effects `Vi`.

## Performance on MipNeRF360 

### PSNR

|            | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ---------- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper      | 24.20   | 32.05  | 29.53   | 20.45   | 27.78  | 31.38   | 31.93 | 25.39 | 22.01    | 27.19 |
| Ours       | 23.89   | 33.20  | 29.96   | 20.62   | 27.07  | 31.51   | 32.35 | 24.79 | 21.49    | 27.21 |
| Ours-w/o D | 23.89   | 33.05  | 29.83   | 20.78   | 27.14  | 31.21   | 31.65 | 24.90 | 21.61    | 27.12 |

`Ours - w/o D` means not adding the `--depth_correct` parameter during training.

## Install

We tested on a server configured with Ubuntu 22.04, CUDA 11.8. 

1. Clone this repo:

```
git clone --recursive https://github.com/LiYukeee/sort-free-gs.git 
cd sort-free-gs
```

2. Install dependencies

```
conda env create --file environment.yml
conda activate sortfree
```

## Data

First, create a ``data/`` folder inside the project path by

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

The MipNeRF360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/).

## Training

In our repository, the parameter settings for `train.py`, `render.py`, and `metrics.py` are identical to those of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 

We have added a new parameter `--depth_correct` to `train.py`, which controls the effect of Equation 10 in the paper. If this parameter is included, the weight calculation is performed as follows:

$$
w(d_i) = \max\left(0, 1 - \frac{d_i}{\sigma}\right) v_i,
$$

which is consistent with the original paper. If the parameter is not included, the weight calculation change to:

$$
w(d_i) = v_i,
$$

This modification allows us to experiment with the impact of depth correction on the rendering process and compare the results with the original model.

### MipNeRF360

```shell
bash train_mipnerf360.sh
```

### Single Scene

```shell
python train.py --eval -s <path to COLMAP or NeRF Synthetic dataset> -m <path to output> --depth_correct
python render.py -m <path to output>
python metrics.py -m <path to output>
```

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).
