# Object-centric Learning with Slot Attention

## Description

This repository contains scripts for training and evaluating an object-centric learning model using Slot Attention. The key scripts included are:

1. `dataset.py` - Defines the dataset class for loading and processing images.
2. `train.py` - Script for training the Slot Attention model.
3. `eval.py` - Script for evaluating the trained model and visualizing the results.

## Setup and Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- PIL
- OpenCV

### Installation

1. Clone this repository 
 
 Ensure you have the necessary directories and data in place:

Dataset should be placed in C:/1projects/codes/Object_centric/data/Choracic/images/.
Usage
1. Dataset Script: dataset.py
Defines the PARTNET dataset class for loading and processing images. It includes transformations and resizing operations to prepare images for training and evaluation.

2. Training Script: train.py
This script trains the Slot Attention model on the specified dataset. It includes various hyperparameters and settings for training:

--model_dir: Directory to save the trained model.
--seed: Random seed for reproducibility.
--batch_size: Batch size for training.
--num_slots: Number of slots in Slot Attention.
--num_iterations: Number of attention iterations.
--hid_dim: Hidden dimension size.
--learning_rate: Learning rate for the optimizer.
--warmup_steps: Number of warmup steps for the learning rate.
--decay_rate: Rate for learning rate decay.
--decay_steps: Number of steps for learning rate decay.
--num_workers: Number of workers for loading data.
--num_epochs: Number of training epochs.


To train the model, 
first put pretrained dino weight in the /tmp folder, which can be downloaded here: "https://drive.google.com/drive/folders/1cl6J09TgDP3sdsvmSIWoicaBBanSobQe?usp=sharing"
run:
python train.py --model_dir ./tmp/model11.pth --num_epochs 1000


To evaluate the model, run:
python eval.py


same train/eval procedure applies to train_dino_x.py  and eval_dino.py