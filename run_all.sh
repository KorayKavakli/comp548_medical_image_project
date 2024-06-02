#!/bin/bash

# This script is used to train all the models in the models folder
python3 -W ignore main.py --settings './settings/alexnet_settings.txt' --mode 'train'
python3 -W ignore main.py --settings './settings/alexnet_settings.txt' --mode 'test'

python3 -W ignore main.py --settings './settings/resnet_settings.txt' --mode 'train'
python3 -W ignore main.py --settings './settings/resnet_settings.txt' --mode 'test'

python3 -W ignore main.py --settings './settings/vgg_settings.txt' --mode 'train'
python3 -W ignore main.py --settings './settings/vgg_settings.txt' --mode 'test'