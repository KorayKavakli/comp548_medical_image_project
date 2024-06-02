# Exploring Convolutional Neural Networks for Dermatological Lesion Classification: Techniques and Performance Analysis

[Koray Kavaklı](https://scholar.google.com.tr/citations?user=rn6XtO4AAAAJ&hl=en&oi=ao/),
[Arda Gülersoy](https://scholar.google.com/citations?user=Q7ImQTkAAAAJ&hl=en/),
[Doğukan Yaprak](https://scholar.google.com/citations?user=7HMgSzoAAAAJ&hl=en&oi=ao/)


## Description
This repository contains our implementation for **COMP548 Medical Image Analysis** *class project*. 
In our work, we evaluated the performance of different neural network architectures (AlexNet, ResNet18, VGG11)for cancerous lesion type classification.
We took advantage of pretrained model architectures and modified their classification layer with respect to our dataset.

### Requirements
We provide all dependecies in `requirements.txt`. 
To install the required packages use the following:
```shell
pip3 install -r requirements.txt
```

## Dataset
In our work we relied on [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/), which can be downloaded from the website. 

Create folders for dataset in following manner:
```
mkdir ./dataset
mkdir ./dataset/HAM10000_images
```
Place all images from dataset to folder `./dataset/HAM10000_images` and 
put the `HAM10000_metadata.csv` into `./dataset/HAM10000_images`

Your folder structure should look like this:
```
└──── <project directory>/
    ├──── dataset/
    |   ├──── HAM10000_images/
    |   |   ├──── ISIC_0024306.jpg/
    |   |   ├──── ...
    |   └──── HAM10000_metada.csv
    ├──── dataloader/
    |   ├──── __init__.py
    |   ├──── ...
    ├──── models/
    |   ├──── __init__.py
    |   ├──── ...
    ├──── settings/
    |   ├──── test-easy.txt
    |   ├──── ...
    ├──── utils/
    |   ├──── __init__.py
    |   ├──── ...
    └──── output/
        ├──── AlexNet/
        ├──── ResNet18/
        └──── VGG11/
```
## Evaluate a sinlge model

### Training
In order to train model with `'./settings/alexnet_settings.txt'`, run the folllowing:

```shell
python3 --settings './settings/alexnet_settings.txt' --mode 'train'
```

### Test
To evaluate the trained model above you can simply use:

```shell
python3 --settings './settings/alexnet_settings.txt' --mode 'test'
```

## Run all models
If you would like to replicate all results for all models we provide a simple bash script. 
It automatically puts corresponding results, plots, metrics and model weights in `./output` folder.

```shell
./run_all.sh
```

### Settings
If you would like to replicate the same results that we obtained we have provided our setting files in `./settings`. 
However if you wish to utilize different parameters please use the `./settings/sample_settings.txt`. 
A sample setting config file is shared below. Please adjust  

```json
{
    "general"                 : {
                                 "device"                : "cuda",
                                 "seed"                  : 108,
                                 "mode"                  : "test",
                                 "output directory"      : "./output"
                                },

    "model"                   : {
                                 "name"                  : "AlexNet",
                                 "normalize"             : 1,
                                 "resize"                : [224, 224],
                                 "mean"                  : [0.485, 0.456, 0.406],
                                 "std"                   : [0.229, 0.224, 0.225],
                                 "num classes"           : 7
                                },

    "hyperparameters"         : {
                                 "epochs"                : 5,
                                 "learning rate"         : 4e-4,
                                 "batch size"            : 128,
                                 "num workers"           : 8
                                },

    "dataset"                 : {
                                 "dataset directory"     : "./dataset",
                                 "images foldername"     : "HAM10000_images",
                                 "csv filename"          : "HAM10000_metadata.csv",
                                 "split ratios"          : [0.8, 0.1, 0.1]
                                 },

    "lesion types"            : {
                                 "nv": 0,
                                 "mel": 1,
                                 "bkl": 2,
                                 "bcc": 3,
                                 "akiec": 4,
                                 "vasc": 5,
                                 "df": 6
                                }
}

```

## Contact
If you have any trouble in replacating this work or have questions please don't hesitate to open an issue in the repository. 




