## FLGAN 

The FLGAN is modified from [[here]](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py) by Erik Linder-Norén et. al.

## Table of Contents
* [Installation](#installation)
* [RunExample](#example-usage)

## Implementations
Implementations of FLGAN and DCGAN based on PyTorch.

* [FLGAN](/implementations/flgan.py)
* [DCGAN](/implementations/dcgan.py)


## Installation

To install dependency, run:

```
$ sudo pip3 install -r requirements.txt
```

## Example Usage 

An example running FLGAN with image size of 64x64, generating images every 500 iterations:

```
$ python3 flgan.py --sample_interval 500 --img_size 64
```

To run with other datasets, please modify the dataloader. See [PyTorch built-in datasets](https://pytorch.org/vision/stable/datasets.html) for details.


