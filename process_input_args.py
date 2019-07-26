#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Dipan S.
# DATE CREATED: 5/20/2019                                  
# PURPOSE: Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:

import argparse


def get_training_input_args():
    
    parser = argparse.ArgumentParser('train')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('--save_dir', default='checkpoints', type = str, help = 'path to the folder to save checkpoint')
    parser.add_argument('--arch', default = 'vgg16', type = str, help = 'CNN model architecture')
    parser.add_argument('--learning_rate', type = float, help = 'Learning rate')
    parser.add_argument('--hidden_units', type = int, help = 'Hiddent units')
    parser.add_argument('--epochs', type = int, help = 'Number of epochs')
    parser.add_argument('--gpu', help = 'Flag for using GPU', action='store_true')
    in_arg = parser.parse_args()   
    return in_arg

def get_predict_input_args():
    
    parser = argparse.ArgumentParser('predict')
    parser.add_argument('path_to_image', help='Path to Image to be predicted')
    parser.add_argument('checkpoints', help='Path to a directory where checkpoints are saved')
    parser.add_argument('--top_k', default=5, type = int, help = 'K top most classes')
    parser.add_argument('--cat_name', help = 'Flower category names')
    parser.add_argument('--gpu', help = 'Flag for using GPU', action='store_true')
    in_arg = parser.parse_args()  
    return in_arg