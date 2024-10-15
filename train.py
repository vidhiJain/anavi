import os
import torch
import torch.nn as nn
from anp.model import DirDis, VisDirDis, ANP, LinearRegressionModel, EgoVisDis, EgoVisDisPool, Resnet101VisDirDis
from anp.trainer import TrainerCE, TrainerMSE 
from arguments import get_config


config = get_config()
optimizer_class = torch.optim.AdamW

# TODO: replace with importlib from anp.model
if config['model']['classname'] == 'ANP':
    model_class = ANP
elif config['model']['classname'] == 'VisDirDis':
    model_class = VisDirDis
elif config['model']['classname'] == 'DirDis':
    model_class = DirDis
elif config['model']['classname'] == 'LinearRegressionModel':
    model_class = LinearRegressionModel
    optimizer_class = torch.optim.SGD
elif config['model']['classname'] == 'EgoVisDis':
    model_class = EgoVisDis
elif config['model']['classname'] == 'EgoVisDisPool':
    model_class = EgoVisDisPool
elif config['model']['classname'] == 'Resnet101VisDirDis':
    model_class = Resnet101VisDirDis
else:
    print(f"Model {config['model']['classname']} not implemented.")
    

if config['model']['use_regression']:
    criterion = nn.MSELoss()
    if config.get('criterion', None) is not None:
        if config['criterion']['classname'] == 'HuberLoss':
            criterion = nn.HuberLoss(delta = config['criterion']['delta'])
    trainer = TrainerMSE(config, model_class, optimizer_class=optimizer_class, criterion=criterion)
else:
    criterion = nn.CrossEntropyLoss()
    trainer = TrainerCE(config, model_class, optimizer_class=optimizer_class, criterion=criterion)

trainer.init_wandb()
trainer.init_training()
test_loss, test_acc = trainer.eval(trainer.test_loader, 'test')
train_loss, train_acc = trainer.eval(trainer.train_loader, 'train')
trainer.save_checkpoint(
    os.path.join(config['chkpt_dir'], 'final-' + config['chkpt_path']),
    epoch=trainer.epoch, 
    train_loss=train_loss,
    test_loss=test_loss,
    train_acc=train_acc,
    test_acc=test_acc,
)
print(f'''Saving final model...
    epoch={trainer.epoch}, 
    train_loss={train_loss},
    test_loss={test_loss},
    train_acc={train_acc},
    test_acc={test_acc}
    '''
)