import os
import yaml
import pandas as pd
import torch
import torch.nn as nn

from anp.model import *
from anp.trainer import TrainerCE, TrainerMSE 
from arguments import get_config

from anp.model import *
from anp.trainer import TrainerMSE

@torch.no_grad()
def get_scores(name_chkpt, name_confg, model_class, split='all', num_samples=10000, epsilon=0.01, device='cpu', **kwargs):

    chkpt_dir = "/data/vdj/ss/checkpoints/"
    code_dir = "/home/vidhij/ss/anp/configs/"
    
    chkpt_path = f'{name_chkpt}.pth'  
    config_filepath = f'{name_confg}.yaml'

    with open(os.path.join(code_dir, config_filepath), 'r') as f:
        config = yaml.safe_load(f)

    model = model_class(**config['model'])
    # model = model.to(device) 
    chkpt = torch.load(os.path.join(chkpt_dir, chkpt_path), map_location='cpu')
    model.load_state_dict(chkpt['model_state_dict'])
    trainer = TrainerMSE(config, model_class)
    trainer.model = model.to(device)
    model.eval()
    
    logs = {}
    if split == 'train':
        dataloaders = [trainer.train_loader]
    elif split == 'val':
        dataloaders = [trainer.val_loader]
    elif split == 'test':
        dataloaders = [trainer.test_loader]
    else:
        dataloaders = [trainer.train_loader, trainer.val_loader, trainer.test_loader]
    
    for idx, dataloader in enumerate(dataloaders):
        if split == 'all':
            split_name = ['train', 'val', 'test'][idx]
        else:
            split_name = split
        count = 0
        pred = []
        true = []
        for batch in dataloader:
            batch = {key: val.to(device) for key, val in batch.items()}
            count += batch['noise'].shape[0]
            
            out = model(batch)
            output = out.cpu().detach() # if device.contains('cuda') else out
            trueval = batch['noise'].cpu().detach() / 128. # if device.contains('cuda') else batch['noise']
            pred.append(output)
            true.append(trueval) 
            if count > num_samples:
                break
        predicted = torch.cat(pred, axis=0)[:num_samples]
        true_labels = torch.cat(true, axis=0)[:num_samples]
        
        r2_score = trainer.get_r2_score(predicted, true_labels)
        eps_acc = trainer.get_accuracy(predicted, true_labels, epsilon=epsilon)

        logs[split_name] = {
            'r2_score': r2_score,
            'eps_acc': eps_acc, 
            'count': true_labels.shape[0]
        }

    return logs
    


def main(savefile, indices=None):
    to_evaluate = [
        dict(
            name = 'DisLinearReg',
            model_class = LinearRegressionModel,
            name_chkpt =  'best-linear_reg-large_summer-durian-6', #  #'visdirdis_freeze_resnet_mse-large' 
            name_confg =  'linear_reg-large', #
            device="cpu"
        ), 
        dict(
            name = 'DirDisMLPReg',
            model_class = DirDis,
            name_chkpt = 'best-resume_dirdis_mse_large_radiant-firefly-11', #
            name_confg =  'dirdis_mse-large', # 
            device="cuda:2"
        ), 
        dict(
            name = 'PanoVisDirDisReg',
            model_class = VisDirDis,
            name_chkpt = 'best-resume_visdirdis_mse_large-2',
            name_confg = 'visdirdis_mse-large-scratch',
            device="cuda:2"
        ),
        dict(
            name = 'PanoVisDisDirReg-pool-frozen-resnet',
            model_class = VisDirDis,
            name_chkpt = 'best-visdirdis_pool_freeze_resnet_rgb_only_mse-large_driven-tree-17',
            name_confg = 'visdirdis_pool_freeze_resnet_rgb_only_mse-large',
            device="cuda:2"
        ),
        dict(
            name = 'PanoVisDisDirReg-frozen-resnet',
            model_class = VisDirDis,
            name_chkpt = 'best-visdirdis_freeze_resnet_rgb_only_mse-large_clear-resonance-14',
            name_confg = 'visdirdis_freeze_resnet_rgb_only_mse-large',
            device="cuda:2"
        ),
        dict(
            name = 'EgoVisDisReg',
            model_class = EgoVisDis,
            name_chkpt = 'best-egovisdis-resnet50_glad-deluge-181',
            name_confg = 'egovisdis-resnet50',
            device="cuda:2"
        ),
        # dict(
        #     name='Heuristic',
        #     model_class = Heuristic,
        #     device="cpu",
        # )
    ]

    if indices is not None:
        to_evaluate = [to_evaluate[i] for i in indices]

    logs = {}
    for cfg in to_evaluate:
        print('Evaluating:', cfg['name'])
        logs[cfg['name']] = get_scores(**cfg)
        print('Results so far: ', logs)
    print(logs)

    df = pd.DataFrame.from_dict({(i,j): logs[i][j] 
                             for i in logs.keys() 
                             for j in logs[i].keys()},
                            orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index, names=['model', 'split'])
    df['r2_score'] = df['r2_score'].round(4)
    df['eps_acc'] = (df['eps_acc']* 100).round(2) 
    df.to_csv(savefile)
    breakpoint()
    print(f'done! Saved results to {savefile}')


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process a savefile name and a list of numbers.')
    parser.add_argument('savefile', type=str, help='The name of the save file.')
    parser.add_argument('--indices', type=int, default=None, nargs='+', help='A list of model to evaluate.')
    args = parser.parse_args()
    print(f'Savefile name: {args.savefile}')
    print(f'List of model to evaluate: {args.indices}')
    main(args.savefile, args.indices)
