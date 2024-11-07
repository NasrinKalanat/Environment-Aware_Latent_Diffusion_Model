# Environment-Aware Latent Diffusion Models


  
## Requirements
A suitable [conda](https://conda.io/) environment named `ealdm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ealdm
```


# Train your own EALDMs


## Model Training

Logs and checkpoints for trained models are saved to `logs/<START_DATE_AND_TIME>_<config_spec>`.

### Training autoencoder models

Configs for training are provided at `configs/autoencoder`.
Training can be started by running
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/<config_spec>.yaml -t --gpus 0,    
```
where `config_spec` is one of {`autoencoder_kl_8x8x64`(f=32, d=64), `autoencoder_kl_16x16x16`(f=16, d=16), 
`autoencoder_kl_32x32x4`(f=8, d=4), `autoencoder_kl_64x64x3`(f=4, d=3)}.

For training VQ-regularized models, see the [taming-transformers](https://github.com/CompVis/taming-transformers) 
repository.

### Training EALDMs 

In ``configs/latent-diffusion/`` we provide configs. 
Training can be started by running
for conditioned model:

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/stdiff_cin-ldm-vq-f8.yaml -t --gpus 0,
``` 
for unconditioned model:

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/uncond_cin-ldm-vq-f8.yaml -t --gpus 0,
``` 


### Get the pretrained autoencoding models

Running the following script downloads and extracts all available pretrained autoencoding models.   
```shell script
bash scripts/download_first_stages.sh
```

The first stage models can then be found in `models/first_stage_models/<model_spec>`

