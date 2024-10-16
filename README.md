# ANAVI

ANAVI is a framework for "Audio Noise Awareness using Visuals of Indoor environments for NAVIgation". This is presented at 8th Annual Conference on Robot Learning, 2024. 

Website: https://anavi-corl24.github.io


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

# Cite 
```
@article{jain2024anavi,
  author    = {Jain, Vidhi and Veerapaneni, Rishi and Bisk, Yonatan},
  title     = {ANAVI: Audio Noise Awareness using Visuals of Indoor environments for NAVIgation},
  journal   = {8th Annual Conference on Robot Learning},
  year      = {2024},
}
```
# Contact Us
If you have questions or concerns, you can raise an issue or email vidhij@andrew.cmu.edu.
