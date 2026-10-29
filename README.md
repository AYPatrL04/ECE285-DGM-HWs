# Chest X-Ray VAEs

## Environment setup

Tested with Python 3.12, PyTorch 2.13.0, TorchVision 0.28.0, and CUDA 13.2 on Linux.

```bash
conda create -n vae python=3.12 -y
conda activate vae
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install numpy pillow matplotlib pytorch-ignite==0.5.5 scipy jupyterlab ipykernel
python -m ipykernel install --user --name vae --display-name "Python (vae)"
jupyter lab
```

For other platforms, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Select the `Python (vae)` kernel. Set `data_dir` in the dataset cell to the directory
containing `train/`, `val/`, and `test/`, each with `NORMAL/` and `PNEUMONIA/`
subdirectories. The default path uses the dataset in the sibling `VAE` directory:

```python
data_dir = Path("../VAE/archive/chest_xray/chest_xray")
```

Run a notebook from top to bottom. The first FID/IS evaluation downloads the
ImageNet-pretrained Inception-v3 weights to the standard PyTorch cache.

## Notebooks

Each notebook is independent and follows model definition, loss definition,
dataset loading, and training. Training uses 128 × 128 grayscale images, 64 latent
dimensions, batch size 32, 100 epochs, seed 42, and BF16 on CUDA. Validation,
prior sampling, reconstruction MSE, FID, and IS are computed in memory.
Each run displays epoch logs, loss curves, original/reconstructed image pairs,
and prior samples inline, without exporting files or checkpoints.

| Notebook | Implementation |
|---|---|
| [VanillaVAE.ipynb](VanillaVAE.ipynb) | Fully connected VAE with hidden widths 512 and 256, ReLU, and a sigmoid output. |
| [VanillaVAESmall.ipynb](VanillaVAESmall.ipynb) | Fully connected VAE with hidden widths 128 and 128. |
| [ConvVAE.ipynb](ConvVAE.ipynb) | Five convolutional stages with channels 32, 64, 128, 256, 256 and ReLU. |
| [ConvVAELarge.ipynb](ConvVAELarge.ipynb) | Convolutional VAE with channels 64, 128, 264, 544, 544. |
| [LeakyVAE.ipynb](LeakyVAE.ipynb) | ConvVAE with LeakyReLU, using negative slope 0.1. |
| [ResVAE.ipynb](ResVAE.ipynb) | Four residual stages with GroupNorm and SiLU. |

All models use Adam with learning rate 0.001 and KL weight 1, except ResVAE,
which uses learning rate 0.0002 and increases its KL weight to 1 over 10 epochs.
