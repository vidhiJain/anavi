import os
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from anp.model import *
from anp.trainer import TrainerMSE, TrainerCE
from arguments import get_config
import argparse


def analyze_chkpt(chkptfile, configfile, device):
    chkpt = torch.load(chkptfile, map_location=device)
    for key in ['run_id', 'epoch', 'train_loss', 'train_accuracy', 'best_val_accuracy', 'best_val_loss', 'current_lr']:
        print(key, chkpt.get(key, None))

    with open(os.path.join('configs/', configfile), 'r') as f:
        config = yaml.safe_load(f)

    if config['model']['use_regression']:
        trainer_class = TrainerMSE
    else:
        trainer_class = TrainerCE

    if config['model']['classname'] == 'ANP':
        model_class = ANP
    elif config['model']['classname'] == 'VisDirDis':
        model_class = VisDirDis
    elif config['model']['classname'] == 'DirDis':
        model_class = DirDis
    elif config['model']['classname'] == 'LinearRegressionModel':
        model_class = LinearRegressionModel
    elif config['model']['classname'] == 'EgoVisDis':
        model_class = EgoVisDis
    elif config['model']['classname'] == 'EgoVisDisPool':
        model_class = EgoVisDisPool
    elif config['model']['classname'] == 'Resnet101VisDirDis':
        model_class = Resnet101VisDirDis
    else:
        print(f"Model {config['model']['classname']} not implemented.")
        

    trainer = trainer_class(config, model_class)
    trainer.load_checkpoint(chkptfile, device=device)
    info = trainer.plot_labels_pred_wrt_distance(trainer.test_loader, step=0)

    dist_arr = info['distances']
    true_arr = info['true']
    pred_arr = info['pred']

    eps_acc = trainer.get_bulk_eps_acc(true_arr, pred_arr)
    breakpoint()
    print(json.dumps(eps_acc, indent=4))

    with open(f"eps_acc/{configfile.split('.')[0]}.json", 'w') as f:
        json.dump(eps_acc, f)

    # Plot the true and predicted labels w.r.t. distance
    plt.clf()
    plt.figure(figsize=(5, 4), dpi=300) 
    plt.tight_layout()
    # plt.yscale('log')
    plt.scatter(dist_arr * 10, true_arr, s=1.0, marker='x')
    plt.scatter(dist_arr * 10, pred_arr, s=1.0, marker='o', c='r')
    # plt.plot(dummy_distances * 10, predicted_db, 'y-', linewidth=4)
    plt.xlabel('Distance (in meters)') # (1 -> 10 m)')
    plt.ylabel('Max Acoustic Noise (in dB)') # (1 -> 128 dB)')
    # plt.title('best fit for distance decibels with DirDis model')
    plt.savefig(f'plot_db_dist/{configfile[:-4]}pdf') 



# Parse command line arguments
parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--chkptfile', type=str, help='Path to checkpoint file')
parser.add_argument('--configfile', type=str, help='Path to config file')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to use (cuda:0 for GPU or cpu for CPU)')

args = parser.parse_args()

# Get the paths from command line arguments
chkptfile = args.chkptfile
configfile = args.configfile
device = args.device

analyze_chkpt(chkptfile, configfile, device)