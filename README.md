# Transfer Learning for DCC
A deep learning project implementing transfer learning with ResNet18 for binary classification of dogs and cats images.

### Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Training](#training)
- [Inference](#nference)
- [Acknowledgments](#acknowledgments)


## 
### Overview:
| Model | Transfer Learning | Datasets |
| :---: | :---:             | :---:    |
| Resnet18 | [train.py](/train.py) | [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data?select=test1.zip)

##
### Installation

Environment Setup
```bash
conda create -n dcc python=3.11
conda activate dcc
git clone https://github.com/k19tvan/DCC_Basic
cd DCC_Basic
pip install requirements.txt
```

Dataset Setup
```bash
gdown https://drive.google.com/uc?id=1MGSHiGYqSPOS692uQt2rRCV2OlCR1rWo
gdown https://drive.google.com/uc?id=1RL_CGNEsHbUErsoxmgOEKToGGRxmDof2
sudo apt install unzip
unzip -o train.zip -d train
unzip -o test.zip -d test
python folder.py
```

##
### Training
Training with Default Hyperparameters
```python
python train.py 
```

Training with Custom Hyperparameters
```python
Example: python train.py --batch_size 100 --num_epochs 20 --freeze_params False ...
```

>The best performing model will be automatically saved in the ./run directory.
##
### Inference
Predict an Image
```python
python test.py --model_path [model_path] --img_path [img_path]
```

```python
Example: python test.py --model_path weight.pth --img_path test/test/1.png
```

##
### Acknowledgments

> Dogs vs. Cats dataset provided by [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats)

>ResNet18 architecture from [torchvision](https://pytorch.org/vision/stable/models.html)