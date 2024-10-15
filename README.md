# ANAVI: Audio Noise Awareness using Visuals of Indoor environments for NAVIgation


# Setup 
1. Download or clone this repo
1. Creat conda/mamba environment 
1. Install `anp` by:
```pip install -e .```
1. Install [PyTorch](https://pytorch.org/get-started/locally/)


# Generate Data
Follow instructions on [DATAGEN.md](./andgen/DATAGEN.md).
Alternately, you can also download the simulated data sample [here]().

# Run
1. Train 
```
python train.py --config=dirdis_huberloss.yaml
python train.py --config=ego_huberloss.yaml
python train.py --config=pano_huberloss.yaml
```
You can train your own model by modifying `config_template.yaml`.

1. Evaluate 
```python train.py --config=configs/visdirdis_ce.yaml```
