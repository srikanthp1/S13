# Image classification - CIFAR add.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/), 
[PyTorch](https://pytorch.org/), 
[torchvision](https://github.com/pytorch/vision) 0.8, 
Uses [matplotlib](https://matplotlib.org/)  for ploting accuracy and losses.

## Info

 * we are training our custom resnet model on pascal_voc dataset. 
 * we are using a custom YOLO which includes skip connections etc
 * Implemented in pytorch 
 * in this repo, i am exporing pytorch-lightning. by using this model that we train i am going to publish space app.
 * this repo only does training using pytorch-lightning


## About

* transforms.py contains the transforms we used using Albumentation library
* dataset.py has dataset class for applying transforms 
* find_LR.py contains a function which fetches max_lr we can use for onecyclelr
* one_cycle_lr returns us one_cycle scheduler
* but we can save lrs in text file and load optimizer and weights every time we restart
* model.py has pytorch settings 
* utils.py has some graph functions and others 
* notebook has all pytorch_lightning results 
* i chose random module to achieve 75% mosiac
* i trained for 20 epochs and made lr constanst for another 10 with which i achieved following accuracy
* training was done on kaggle so i attached two notebooks one i trained with pytorchlightning and the other with torch
* training would take about 14 hr for 25 epchs 

## Results 


### accuracy 

 * Class accuracy is: 73.040520%
 * No obj accuracy is: 97.752090%
 * Obj accuracy is: 69.856880%

## Usage

```bash
git clone https://github.com/srikanthp1/S13.git
```
* notebook for pytorch_lightning


## Model details

```python
model = model().to(device)
summary(model, input_size=(3, 32, 32))
```

## Analysis 

* accuracy drop was gradual 
* stopped progres at 5.55 but still kept training for 5 more epochs 
* did training with both torch one and pytorchlightning one and somehow for the given numbe rof epochs both gave 75,95,70
* but more training can yield better results perhaps 40 epochs would have done it is the conclusion by loss analysis


