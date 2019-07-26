#!/bin/sh
# */ImageClassifier/run_bulk_training.sh
#                                                                             
# PROGRAMMER: Dipan S.
# DATE CREATED: 05/29/2019                                  
# REVISED DATE: 05/29/2019  - 
# PURPOSE: Runs bulk training for mulitple ImageNet pretrained architectures
#
# Usage: sh run_bulk_training.sh    -- will run program from commandline within Project Workspace
# 

python train.py flowers --arch alexnet  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
python train.py flowers --arch vgg11  --epoch 12 --hidden_units 4096 --save_dir checkpoints --gpu
python train.py flowers --arch vgg13  --epoch 12 --hidden_units 4096 --save_dir checkpoints --gpu
python train.py flowers --arch vgg16  --epoch 12 --hidden_units 4096 --save_dir checkpoints --gpu
python train.py flowers --arch vgg19  --epoch 12 --hidden_units 4096 --save_dir checkpoints --gpu
python train.py flowers --arch resnet18  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
python train.py flowers --arch resnet34  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
#python train.py flowers --arch resnet50  --epoch 1 --hidden_units 512 --save_dir checkpoints
python train.py flowers --arch resnet50  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
python train.py flowers --arch resnet101  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
python train.py flowers --arch resnet152  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
python train.py flowers --arch densenet121  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
python train.py flowers --arch densenet161  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
python train.py flowers --arch densenet169  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu
python train.py flowers --arch densenet201  --epoch 10 --hidden_units 512 --save_dir checkpoints --gpu