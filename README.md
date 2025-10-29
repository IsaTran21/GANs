# Generative Adversarial Networks (GANs) with PyTorch

A PyTorch-based implementation of popular **Generative Adversarial Networks (GANs)** including:
- **ProGAN** (Progressive Growing of GANs)
- **CycleGAN** (Image-to-Image Translation)

## Overview

This repository provides clean and modular implementations of **GAN architectures** from scratch using **PyTorch**.  
It aims to serve as both a **learning resource** and a **baseline framework** for researchers and developers exploring generative models.

Each GAN is trained from scratch and supports:
- Configurable architectures
- Custom datasets
- Checkpoint saving & resuming
- GPU on Google colab (if there are multiple GPUs, can tailor it to distributed training).
- Logging to show details about data processing issue if any and the save models details
- Using tensorboard to show the images at each resolution.

## Installation
### For the ProGAN
Clone the repository and install dependencies, on Window:

```bash
git init
git clone git@github.com:IsaTran21/GANs.git
pip install -r GANs\requirements.txt
cd GANs\ProGAN
python run_training.py
```

On Jupyter notebook:
[Notebook link](https://drive.google.com/file/d/1j84OzBna1KrpKC1r7BDPqOfAFWXIukd0/view?usp=sharing)
