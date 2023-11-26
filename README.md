# Percieved face reconstruction from fMRI with improved identity consistency
This repository hosts the official PyTorch implementation of the paper: "**Brain-driven facial image reconstruction via StyleGAN inversion with improved identity consistency**"

 * [Paper: coming soon](https://xxx)

## Recent Updates
**`2023.11.26`**: Initial code release 🎉

## Getting Started
### Prerequisites
```bash
$ torch==2.0.1
```
Checking scripts/test.py for demo case.
### Pretrained Models
Please download the pre-trained models from the following links. fMRI2face model contains the entire fMRI-driven face generation architecture, including the encoder and decoder weights. If you wish to use the pretrained model for inference, you may do so using the flag `--checkpoint_path`.
| Path | Description
| :--- | :----------
|[fMRI2face](https://drive.google.com/file/d/1h2QZ2wVaEpVnZ_Oay5bEF3fa3WrGXC8F/view?usp=sharing)  | brain-driven face reconstruction model with fMRI data of subject 1.


In addition, we provide the links of various auxiliary models needed for training your own fMRI2face model from scratch.
| Path | Description
| :--- | :----------
|[StyleGAN Inversion](https://github.com/wty-ustc/e2style)  | StyleGAN inversion model trained with the FFHQ dataset taken from [here](https://github.com/wty-ustc/e2style).

| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://github.com/rosinality/stylegan2-pytorch) | StyleGAN model pretrained on FFHQ taken from [here](https://github.com/rosinality/stylegan2-pytorch).


By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 
    
### Training 
TODO: Scripts for training and data preprocess will be released when they are ready.

## Testing
### Inference
Having trained  model, you can use `scripts/test.py` to apply the model on a set of [test data (download link)](https://drive.google.com/file/d/1rPZgPTwGfes5J4eldheFKtS_CMz1kpqx/view?usp=sharing).

## Acknowledgements
This code is heavily based on [e2style](https://github.com/wty-ustc/e2style).

The full fMRI data sets for all four subjects can be available from [VanRullen19 dataset (download link)](https://openneuro.org/datasets/ds001761/versions/2.0.1).

## Citation

coming soon

