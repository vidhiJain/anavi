import argparse
import json
import os
import torch
import yaml


def get_config():
    # Create the parser and add argument
    parser = argparse.ArgumentParser(description='Load YAML config.')
    parser.add_argument('--config', type=str, default='visdirdis_ce.yaml', 
        help='Path to the YAML config file')
    parser.add_argument('--chkpt_path', type=str, default=None,
        help="Path to the checkpoint file")
    parser.add_argument('--regression', 
        help="Regression task", default=None, type=bool)
    parser.add_argument('--resume', 
        help="Resume training", default=None, type=bool)
    parser.add_argument('--cuda', type=int, default=0, 
        help="check the available GPU that you want to run this script.")
    args = parser.parse_args()

    # Load the config from the YAML file
    with open(os.path.join('configs/', args.config), 'r') as f:
        config = yaml.safe_load(f)

    config['resume_training'] = args.resume if args.resume is not None else config['resume_training']
    config['model']['use_regression'] = args.regression if args.regression is not None else config['model']['use_regression']
    config['chkpt_path'] = args.chkpt_path if args.chkpt_path is not None else config['chkpt_path']
    config['device'] = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    # Now you can access your config parameters as dictionary
    print(json.dumps(config, indent=4))
    return config