# ERA5 Autoencoder

## Overview

This project implements a **distributed autoencoder** designed to process weather data using **PyTorch's DistributedDataParallel (DDP)**. It supports the following modes:

- ✅ **Training from scratch** (`--train`)
- 🔁 **Continuing training from a saved model** (`--start_from_model`)
- 🔍 **Inference only** (`--infer`)

The autoencoder works with **ERA5 weather data in Zarr format** and supports three variables:

- 🌡️ `t2m`: 2m temperature  
- ☀️ `ssrd`: Surface solar radiation  
- 🌧️ `tp`: Total precipitation

---

## Requirements

### Hardware
- ⚙️ NVIDIA GPU(s) recommended for optimal performance  
- 💾 Minimum 32GB RAM for full dataset processing

### Software Dependencies

- Python 3.11.11
- `torch==2.6.0` (with CUDA support)
- `numpy==1.26.4`
- `xarray==2024.3.0`
- `zarr==2.18.3`
- `dask==2025.1.0`

### Installation

Install dependencies via pip:

```bash
pip install -e .
```
### Usage Examples

1. 🛠️ Train a new autoencoder on temperature data

```bash

python autoencoder_run.py --train \
  --input_zarr /data/era5.zarr \
  --variable_type t2m \
  --model_output t2m_model.pth \
  --output_zarr t2m_latents.zarr \
  --batch_size 32768 \
  --num_workers 4
```

2. 🔁 Continue training a saved model

```bash

python autoencoder_run.py --start_from_model ssrd_model.pth \
  --input_zarr /data/era5.zarr \
  --variable_type ssrd \
  --model_output ssrd_model_finetuned.pth
```

3. 🔍 Inference on precipitation data using a trained model

```bash
python autoencoder_run.py --infer tp_model.pth \
  --input_zarr /data/era5.zarr \
  --variable_type tp \
  --output_zarr tp_latents.zarr

```

## ARGUMENTS NATURE

REQUIRED
Argument | Description
--input_zarr | Path to input Zarr dataset
--variable_type | Variable to process (t2m, ssrd, tp)

OPTIONAL
Argument | Description | Default
--model_output | Path to save trained model | best_model.pth
--output_zarr | Path to save output Zarr | None
--batch_size | Training/inference batch size | 64384
--num_workers | Number of DataLoader workers | 8
--prefetch_factor | Dataloader prefetch factor | 25

### Future Additions 

- OpenEO Compliant UDF addition
- Creating a docker container chained with OpenEo processes
- Extensively use destineEarth ERA5 ZARR collections directly
- Use Raster2stac to directly push the output latent space xarray datasets to the stac catalogue

