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
git clone https://github.com/srikanthp1/S12.git
```
* notebook for pytorch_lightning


## Model details

```python
model = model().to(device)
summary(model, input_size=(3, 32, 32))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         GroupNorm-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
         GroupNorm-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
         GroupNorm-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
         GroupNorm-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
        GroupNorm-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 32, 32]          73,728
        GroupNorm-14          [-1, 128, 32, 32]             256
           Conv2d-15          [-1, 128, 32, 32]         147,456
        GroupNorm-16          [-1, 128, 32, 32]             256
           Conv2d-17          [-1, 128, 32, 32]           8,192
        GroupNorm-18          [-1, 128, 32, 32]             256
       BasicBlock-19          [-1, 128, 32, 32]               0
           Conv2d-20          [-1, 128, 32, 32]         147,456
        GroupNorm-21          [-1, 128, 32, 32]             256
           Conv2d-22          [-1, 128, 32, 32]         147,456
        GroupNorm-23          [-1, 128, 32, 32]             256
       BasicBlock-24          [-1, 128, 32, 32]               0
           Conv2d-25          [-1, 256, 16, 16]         294,912
        GroupNorm-26          [-1, 256, 16, 16]             512
           Conv2d-27          [-1, 256, 16, 16]         589,824
        GroupNorm-28          [-1, 256, 16, 16]             512
           Conv2d-29          [-1, 256, 16, 16]          32,768
        GroupNorm-30          [-1, 256, 16, 16]             512
       BasicBlock-31          [-1, 256, 16, 16]               0
           Conv2d-32          [-1, 256, 16, 16]         589,824
        GroupNorm-33          [-1, 256, 16, 16]             512
           Conv2d-34          [-1, 256, 16, 16]         589,824
        GroupNorm-35          [-1, 256, 16, 16]             512
       BasicBlock-36          [-1, 256, 16, 16]               0
           Conv2d-37            [-1, 512, 8, 8]       1,179,648
        GroupNorm-38            [-1, 512, 8, 8]           1,024
           Conv2d-39            [-1, 512, 8, 8]       2,359,296
        GroupNorm-40            [-1, 512, 8, 8]           1,024
           Conv2d-41            [-1, 512, 8, 8]         131,072
        GroupNorm-42            [-1, 512, 8, 8]           1,024
       BasicBlock-43            [-1, 512, 8, 8]               0
           Conv2d-44            [-1, 512, 8, 8]       2,359,296
        GroupNorm-45            [-1, 512, 8, 8]           1,024
           Conv2d-46            [-1, 512, 8, 8]       2,359,296
        GroupNorm-47            [-1, 512, 8, 8]           1,024
       BasicBlock-48            [-1, 512, 8, 8]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 27.00
Params size (MB): 42.63
Estimated Total Size (MB): 69.64
----------------------------------------------------------------

```

## Analysis 

* accuracy drop was gradual 
* stopped progres at 5.55 but still kept training for 5 more epochs 
* did training with both torch one and pytorchlightning one and somehow for the given numbe rof epochs both gave 75,95,70
* but more training can yield better results perhaps 40 epochs would have done it is the conclusion by loss analysis


