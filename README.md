# Unofficial Implementation of Sort-Free Gaussian Splatting

https://arxiv.org/pdf/2410.18931

This is an unofficial implementation of the ICLR'25 paper "Sort-free Gaussian Splatting via Weighted Sum Rendering". [Paper](https://arxiv.org/pdf/2410.18931)

## Performance on Mip-Nerf360

### PSNR

|       | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ----- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper | 24.20   | 32.05  | 29.53   | 20.45   | 27.78  | 31.38   | 31.93 | 25.39 | 22.01    | 27.19 |
| Ours  |         |        |         |         |        |         |       |       |          |       |

### SSIM

|       | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ----- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper | 0.744   | 0.938  | 0.909   | 0.580   | 0.872  | 0.923   | 0.925 | 0.728 | 0.614    | 0.804 |
| Ours  |         |        |         |         |        |         |       |       |          |       |

### LPIPS

|       | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ----- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper | 0.205   | 0.199  | 0.191   | 0.342   | 0.097  | 0.125   | 0.197 | 0.235 | 0.311    | 0.211 |
| Ours  |         |        |         |         |        |         |       |       |          |       |

## Install

We tested on a server configured with Ubuntu 22.04, cuda 11.8. 

1. Clone this repo:

```
git clone https://github.com/ --recursive
cd Sortfree-gs
```

2. Install dependencies

```
conda env create --file environment.yml
conda activate Sortfree-gs
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

### Mip-Nerf360

```shell
bash train_mipnerf360.sh
```

### Single Scene

```shell
python train.py --eval -s <path to COLMAP or NeRF Synthetic dataset> -m <path to output> --depth_correct
```

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).
