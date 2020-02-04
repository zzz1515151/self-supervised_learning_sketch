# Deep Self-Supervised Representation Learning for Free-Hand Sketch

![](https://img.shields.io/badge/language-Python-{green}.svg)
![](https://img.shields.io/npm/l/express.svg)

<div align=center><img src="https://github.com/zzz1515151/self-supervised_learning_sketch/blob/master/img/cat.gif" width = 30% height = 30% /></div>

This repository is the official code of paper ["Deep Self-Supervised Representation Learning for Free-Hand Sketch"](https://arxiv.org/pdf/2002.00867.pdf)

<div align=center><img src="https://github.com/zzz1515151/self-supervised_learning_sketch/blob/master/img/pipeline.png"/></div>

## Requirements
Ubuntu 16.04

Anaconda 4.5.4 or higher

Python 3.6

PyTorch 0.4.1 

Our hardware environment: 2 Intel(R) Xeon(R) CPUs (E5-2620 v3 @ 2.40GHz), 128 GB RAM, 4 GTX 1080 Ti GPUs.

All the following codes can run on single GTX 1080 Ti GPU.

## Prepare Data 
We use [**Quick, Draw**](https://github.com/googlecreativelab/quickdraw-dataset) dataset for our experiment. The dataset is splited into pretraining dataset and retrevial dataset by 10,000 samples per class and 1,100 samples per class. Image datasets and stroke datasets are used for traning CNN and TCN.

You can use script 'download.sh' to get all these datasets above. Pretraining dataset, query dataset and gallery dataset should be stored separately.

## Setup
```
# 1. Choose your workspace and download our repository.
cd ${CUSTOMIZED_WORKSPACE}
git clone https://github.com/zzz1515151/self-supervised_learning_sketch

# 2. Enter this repository.
cd self-supervised_learning_sketch

# 3. Clone our environment, and activate it.
conda-env create --name ${CUSTOMIZED_ENVIRONMENT_NAME} --file ./conda_environment.yml
conda activate ${CUSTOMIZED_ENVIRONMENT_NAME}

# 4. Download our image and stroke pretraining datasets
chmod +x download.sh
./download.sh
```

## Training Rotation CNN
We follow ["Unsupervised Representation Learning by Predicting Image Rotations"](https://arxiv.org/pdf/1803.07728.pdf) to build RotNet. 
```
# 1. Enter config folder.
cd config 

# 2. Set dataset path or other settings in 'cnn_rotation.py'.
dataset_root = ${CUSTOMIZED_IMAGE_PRETRAIN_ROOT}

# 3. Run script.
cd ..
python train_cnn.py 
    --exp cnn_rotation
    --num_workers ${CUSTOMIZED_NUM_WORKERS} 
    --gpu ${CUSTOMIZED_GPU_ID}

```

## Training Deformation CNN
```
# 1. Enter config folder.
cd config 

# 2. Set dataset path or other settings in 'cnn_deform.py'. 
dataset_root = ${CUSTOMIZED_IMAGE_PRETRAIN_ROOT}

# 3. Run script.
cd ..
python train_cnn.py 
    --exp cnn_deform
    --num_workers ${CUSTOMIZED_NUM_WORKERS} 
    --gpu ${CUSTOMIZED_GPU_ID}

```

## Training Rotation TCN 
We build our TCN network based on following settings:

<div align=center><img src="https://github.com/zzz1515151/self-supervised_learning_sketch/blob/master/img/TCN.png"/></div>


```
# 1. Enter config folder.
cd config 

# 2. Set dataset path or other settings in 'tcn_rotation.py'. 
dataset_root = ${CUSTOMIZED_STROKE_PRETRAIN_ROOT}

# 3. Run script.
cd ..
python train_tcn.py 
    --exp tcn_rotation
    --num_workers ${CUSTOMIZED_NUM_WORKERS} 
    --gpu ${CUSTOMIZED_GPU_ID}

```

## Training Deformation TCN 
```
# 1. Enter config folder.
cd config 

# 2. Set dataset path or other settings in 'tcn_deform.py'. 
dataset_root = ${CUSTOMIZED_STROKE_PRETRAIN_ROOT}

# 3. Run script.
cd ..
python train_tcn.py 
    --exp tcn_deform
    --num_workers ${CUSTOMIZED_NUM_WORKERS} 
    --gpu ${CUSTOMIZED_GPU_ID}

```
## Citations
If you find this code useful to your research, please cite our paper as the following bibtex:
```
@article{selfsupervisedsketch2020,
  title={Deep Self-Supervised Representation Learning for Free-Hand Sketch},
  author={Xu, Peng and Song, Zeyu and Yin, Qiyue and Song, Yi-Zhe and Wang, Liang},
  journal={arXiv preprint arXiv:2002.00867},
  year={2020}
}
```