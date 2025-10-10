# Sort-free Gaussian Splatting via Weighted Sum Rendering



This repository is a re-implementation of the ICLR'25 paper "Sort-free Gaussian Splatting via Weighted Sum Rendering". [Paper](https://arxiv.org/pdf/2410.18931).

We are honored that this re-implementation has been recognized and linked by the original author, [Qiqi Hou](https://hqqxyy.github.io/).

Our implementation builds upon the original 3DGS by incorporating global parameters `sigma` and `weight_background`, as well as a set of `sh` parameters for each GS to decode the view-dependent effects `Vi`.

## Performance on MipNeRF360 

### PSNR

|            | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ---------- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper      | 24.20   | 32.05  | 29.53   | 20.45   | 27.78  | 31.38   | 31.93 | 25.39 | 22.01    | 27.19 |
| Ours       | 23.89   | 33.20  | 29.96   | 20.62   | 27.07  | 31.51   | 32.35 | 24.79 | 21.49    | 27.21 |
| Ours-w/o D | 23.89   | 33.05  | 29.83   | 20.78   | 27.14  | 31.21   | 31.65 | 24.90 | 21.61    | 27.12 |

### SSIM

|            | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ---------- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper      | 0.744   | 0.938  | 0.909   | 0.580   | 0.872  | 0.923   | 0.925 | 0.728 | 0.614    | 0.804 |
| Ours       | 0.714   | 0.951  | 0.919   | 0.581   | 0.850  | 0.931   | 0.930 | 0.708 | 0.620    | 0.800 |
| Ours-w/o D | 0.714   | 0.948  | 0.918   |	0.582 	| 0.851  |0.931    | 0.929 |0.713  | 0.622    | 0.801 |

### LPIPS

|            | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ---------- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper      | 0.205   | 0.199  | 0.191   | 0.342   | 0.097  | 0.125   | 0.197 | 0.235 | 0.311    | 0.211 |
| Ours       | 0.265   | 0.144  | 0.176   | 0.329   | 0.128  | 0.117   | 0.190 | 0.275 | 0.347    | 0.219 |
| Ours-w/o D | 0.265   | 0.175  | 0.176   | 0.329   | 0.128  | 0.117   | 0.190 | 0.272 | 0.344    | 0.222 |


`Ours - w/o D` means not adding the `--depth_correct` parameter during training.

We have added a new parameter `--depth_correct` to `train.py`, which controls the effect of Equation 10 in the paper. If this parameter is included, the weight calculation is performed as follows:

$$
w(d_i) = \max\left(0, 1 - \frac{d_i}{\sigma}\right) v_i
$$

which is consistent with the original paper. If the parameter is not included, the weight calculation change to:

$$
w(d_i) = v_i
$$

This modification allows us to experiment with the impact of depth correction on the rendering process and compare the results with the original model.

## Install

We tested on a server configured with Ubuntu 22.04, CUDA 11.8. 

1. Clone this repo:

```
git clone --recursive https://github.com/LiYukeee/sort-free-gs.git --recursive
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
