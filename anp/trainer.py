from abc import ABC, abstractmethod
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from anp.model import VisDirDis
from anp.utils import get_accuracy, make_dataloaders, convert_to_bins, bin_to_centroid

# Fix seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer(ABC):
    def __init__(self, config, model_class, optimizer_class=optim.AdamW, criterion=nn.CrossEntropyLoss(), scheduler_class=optim.lr_scheduler.StepLR):
        self.config = config
        self.eps_checklist = [0.001, 0.007, 0.009, 0.01, 0.02, 0.025, 0.03, 0.07, 0.1, 0.3, 0.5, 1.0]

        self.device = self.config["device"]
        self.train_loader, self.val_loader, self.test_loader = make_dataloaders(**self.config['data'])
        self.model = model_class(**self.config['model']).to(self.device)
        if len(list(self.model.parameters())):
            self.optimizer = optimizer_class(self.model.parameters(), lr=self.config['optim']['lr'])
            self.scheduler = scheduler_class(self.optimizer, **self.config['scheduler'])
        self.criterion = criterion
        if self.config['resume_training']:
            self.start_epoch, self.run_id = self.resume_training()
            self.resume = "must"
        else:
            self.start_epoch = -1
            self.run_id = None
            self.resume = "never"
            self.best_val_loss = None
            self.best_val_accuracy = None
        # # Init wandb and training
        # self.init_wandb()
        # self.init_training()
        
    def get_learning_rate(self, epoch):
        # Print the current learning rate
        for param_group in self.optimizer.param_groups:
            print(f"Epoch {epoch+1}, Learning Rate: {param_group['lr']}")
        
        return param_group['lr']
    
    def save_checkpoint(self, path, **kwargs):
        # Check if path exists
        path = f"{path.split('.')[0]}_{wandb.run.name}.pth"
        print(f'Updating path for saving checkpoint at {path}.')
        # Save the checkpoint
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            'run_id': wandb.run.id, 
            **kwargs
        }, path)
        print(f'Saved checkpoint at {path}.')
        wandb.save(path, policy='end')

    def init_wandb(self):
        # Initialize wandb
        wandb.init(
            **self.config['wandb'],
            config=self.config, 
            id=self.run_id, 
            resume=self.resume
        )
        print("wandb.run.id = ", wandb.run.id)
        print("wandb.run.name = ", wandb.run.name)

    def init_training(self):
        # Training loop with early stopping
        self.best_val_loss = float('inf') if self.best_val_loss is None else self.best_val_loss
        self.best_val_accuracy = 0.0 if self.best_val_accuracy is None else self.best_val_accuracy
        self.no_improvement_count = 0
        epochs = self.config['epochs']
        patience = self.config['patience']
        for epoch in range(self.start_epoch+1, epochs):
            self.epoch = epoch
            train_loss, train_accuracy = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate()
            
            # Save latest checkpoint
            self.save_checkpoint(
                os.path.join(self.config['chkpt_dir'], 'latest-' + self.config['chkpt_path']), 
                epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                best_val_accuracy=self.best_val_accuracy,
                best_val_loss = self.best_val_loss,
                current_lr=self.scheduler.get_last_lr()[0],
            )
            
            # # Step the scheduler
            # self.scheduler.step()
            
            # Log average losses and lr
            lr = self.get_learning_rate(epoch)
            if wandb.run is not None:
                wandb.log({
                    'Epoch-end/Train Loss': train_loss,
                    'Epoch-end/Val Loss': val_loss,
                    'Train Accuracy': train_accuracy,
                    'Val Accuracy': val_accuracy,
                    'Learning Rate': lr,
                })
            print(f'''Epoch [{epoch+1}/{epochs}], Learning Rate: {lr:.6f}
                  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, 
                  Train Accuracy: {train_accuracy * 100:.3f}%, Val Accuracy: {val_accuracy* 100:.3f}%,
                  ''')

            # Log Gradient diagnostics 
            if wandb.run is not None:
                self.log_gradient_histograms(wandb.run.step)
                self.log_gradient_norms(wandb.run.step)
                self.plot_grad_flow(self.model.named_parameters(), wandb.run.step)

            # Early stopping check
            stop = self.early_stopping(val_loss, patience)
            
            # Save the best checkpoint xif any
            if self.best_val_accuracy < val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss

                self.save_checkpoint(
                    os.path.join(self.config['chkpt_dir'], 'best-' + self.config['chkpt_path']), 
                    epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy, 
                    val_loss=val_loss, val_accuracy=val_accuracy, 
                    best_val_accuracy=self.best_val_accuracy,
                    best_val_loss = self.best_val_loss,
                    current_lr=self.scheduler.get_last_lr()[0],
                )

            if stop: 
                break
        print('Finished Training')

        # evaluate on test set
        test_loss, test_accuracy = self.eval()
        if wandb.run is not None:
            wandb.log({'Test Loss': test_loss, 'Test Accuracy': test_accuracy})
        print('Test Loss: ', test_loss, 'Test Accuracy', test_accuracy)

    def load_checkpoint(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def resume_training(self):
        # Load the checkpoint
        path = os.path.join(
            self.config['chkpt_dir'], 
            f"best-{self.config['chkpt_path'].split('.')[0]}_{self.config['prev_wandb_run_name']}.pth"
        )
        if not os.path.exists(path):
            print('Best model checkpoint not found. Loading the latest checkpoint...')
            path = os.path.join(
                self.config['chkpt_dir'], 
                f"latest-{self.config['chkpt_path'].split('.')[0]}_{self.config['prev_wandb_run_name']}.pth"
            )
        print('Path: ', path)
        try:
            checkpoint = torch.load(
                path,
                map_location='cpu'
            )
        except:
            print('Unable to load chkpt. Check if the path is correct.')
            sys.exit(0)

        # Create a copy of previous chkpt. 
        # This is to ensure that the previous chkpt is not overwritten by the new chkpt.
        new_path = path.split('.')[0] + '_backup.pth'
        torch.save(checkpoint, new_path)
        print('Saved a backup at path: ', new_path)
        # Restore the model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        for name, param in self.model.named_parameters():
            if 'feature_extraction' in name.split('.'):
                param.requires_grad = not self.config['model']['freeze_resnets']

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint.keys():
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        wandb_run_id = checkpoint['run_id']
        print(f'Resuming training from epoch {start_epoch} and wandb run {wandb_run_id}')
        for key in checkpoint.keys():
            if key not in ['model_state_dict', 'optimizer_state_dict', 'epoch', 'run_id']:
                setattr(self, key, checkpoint[key])
                print(f'{key} = {checkpoint[key]}')
        return start_epoch, wandb_run_id

    def early_stopping(self, val_loss, patience):
        # Early stopping check
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= patience:
                print("Early stopping triggered")
                return True
        return False

    # Gradient logging functions
    def log_gradient_histograms(self, step, skip_params=[]): #'rgb_net', 'depth_net']):
        gradhistograms = {}
        for name, param in self.model.named_parameters():
            if name.split('.')[0] not in skip_params and param.grad is not None:
                gradhistograms[f'gradhistogram/{name}'] = wandb.Histogram(param.grad.cpu().numpy())
        wandb.log(gradhistograms, step=step)

    def log_gradient_norms(self, step, skip_params=[]): #['rgb_net', 'depth_net']):
        gradient_norms = {f'gradnorms/{name}': param.grad.norm().item() for name, param in self.model.named_parameters() if name.split('.')[0] not in skip_params and param.grad is not None}
        wandb.log(gradient_norms, step=step)

    def plot_grad_flow(self, named_parameters, step, skip_params=['rgb_net', 'depth_net']):
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None and n.split('.')[0] not in skip_params:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
        
        fig, ax = plt.subplots()
        ax.plot(ave_grads, alpha=0.3, color='b', label='average gradient')
        ax.plot(max_grads, alpha=0.3, color='r', label='max gradient')
        ax.hlines(0, 0, len(ave_grads)+1, linewidth=1, color='k')
        ax.set_xticks(range(0, len(ave_grads), 1))
        ax.set_xticklabels(layers, rotation='vertical')
        ax.set_xlim(xmin=0, xmax=len(ave_grads))
        ax.set_xlabel('Layers')
        ax.set_ylabel('Gradient value')
        ax.set_title('Gradient flow')
        ax.grid(True)
        ax.legend()
        wandb.log({"Gradient Flow": wandb.Image(fig)}, step=step)
        plt.close(fig)

    @abstractmethod
    def train_epoch(self, epoch):
        raise NotImplementedError
    
    @abstractmethod
    def validate(self):
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        raise NotImplementedError

    def get_bulk_eps_acc(self):
        raise NotImplementedError


class TrainerCE(Trainer):
    def __init__(self, config, model_class, optimizer_class=optim.AdamW, criterion=nn.CrossEntropyLoss(), scheduler_class=optim.lr_scheduler.StepLR):
        assert config['model']['use_regression'] == False, "Model must be a classification model"
        super().__init__(config, model_class, optimizer_class, criterion, scheduler_class)
        self.label_range = config['model']['label_range']
        self.num_bins = config['model']['num_bins']

    def get_bulk_eps_acc(self, true_arr, binned_pred_arr):
        
        pred_bin_value = bin_to_centroid(binned_pred_arr, self.config['model']['label_range'], self.config['model']['num_bins']).reshape(-1, 1)
        eps_acc = {}
        for eps in self.eps_checklist:
            acc = ((np.abs(true_arr - pred_bin_value)/self.config['model']['label_range']) < eps).sum() / 7500.0 * 100
            eps_acc[eps] = acc.item() #/ 7500.0
        return eps_acc

    def get_accuracy(self, outputs, labels):
        # Get the predicted labels by taking the argmax of the logits along the class dimension
        _, predicted = torch.max(outputs, 1)
        # Compare the predicted labels with the true labels
        correct = (predicted == labels).sum().item()
        # Compute the accuracy as the ratio of correct predictions to total predictions
        accuracy = correct / labels.size(0)
        return accuracy
        
    def get_predictions(self, logit_values):
        return torch.max(logit_values, 1)[1]
    
    def plot_labels_pred_wrt_distance(self, dataloader, step):
        """Runs on forward pass on GPU then plotting on CPU"""
        model = self.model
        device = self.device

        true, pred, dist = [], [], []
        model.eval()
        model = model.to(device)
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            predicted = self.get_predictions(out)
            dist.append(batch['direction_distance'].cpu().detach().numpy()[..., 1])
            true.append(batch['noise'].cpu().detach().numpy())
            pred.append(predicted.cpu().detach().numpy())
            
        dist_arr = np.concatenate(dist)
        pred_arr = np.concatenate(pred)
        true_arr = np.concatenate(true)
        target = convert_to_bins(
            torch.tensor(true_arr), 
            self.config['model']['label_range'], 
            self.config['model']['num_bins'], 
            'cpu'
        ).squeeze()
        if wandb.run is not None:
            plt.clf()
            fig = plt.figure(figsize=(5, 4))
            plt.scatter(dist_arr, target, marker='x', s=2.0, label='true')
            plt.scatter(dist_arr, pred_arr, marker='x', s=2.0, c='r', label='pred')
            plt.xlabel('Distance (in meters)') # (1 -> 10 m)')
            plt.ylabel('Max Acoustic Noise (in dB)') # (1 -> 128 dB)')
            plt.legend()
            wandb.log({"dB by Distance": wandb.Image(fig)}, step=step)
            plt.close(fig)
        return {'distances': dist_arr, 'true': true_arr, 'binned_true': target, 'pred': pred_arr}

    def train_epoch(self, epoch):
        criterion = self.criterion
        train_loader = self.train_loader

        running_train_loss = 0.0
        self.model.train()

        logit_values = []
        true_values = []
        for i, data in enumerate(train_loader):
            inputs = {key: data[key].to(self.device) for key in data if key != 'noise'}
            labels = data['noise'].float().to(self.device)
            # labels = torch.round(torch.clamp(labels, 0, self.config['model']['num_bins']-1)).long().squeeze()
            target = convert_to_bins(labels, self.label_range, self.num_bins, self.device).squeeze()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            assert outputs.shape[-1] == self.num_bins, f"Output shape: {outputs.shape} does not match num_bins: {self.num_bins}"
            loss = criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            # Step the scheduler
            self.scheduler.step()
            
            running_train_loss += loss.item()

            # Log loss to wandb every mini-batch
            if i % self.config['summary_interval'] == 0:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (epoch, i, loss.item()))
                accuracy = self.get_accuracy(outputs, target)
                print('Train Accuracy @ minibatch', accuracy)
                logit_values.append(outputs)
                true_values.append(target)
            
        train_accuracy = self.get_accuracy(
            torch.cat(logit_values, dim=0),
            torch.cat(true_values, dim=0)
        )
        train_loss = running_train_loss/len(train_loader)
        return train_loss, train_accuracy

    def validate(self):
        model = self.model
        criterion = self.criterion
        val_loader = self.val_loader

        model.eval()
        logit_values = []
        true_values = []
        distances = []
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: batch[key].to(self.device) for key in batch if key != 'noise'}
                labels = batch['noise'].float().to(self.device)
                # labels = torch.round(torch.clamp(labels, 0, self.config['model']['num_bins']-1)).long().squeeze()
                target = convert_to_bins(labels, self.label_range, self.num_bins, self.device).squeeze()
                outputs = model(inputs)
                val_loss = criterion(outputs, target)
                running_val_loss += val_loss.item()
                if wandb.run is not None:
                    wandb.log({'Running Val Loss': val_loss.item()})
                logit_values.append(outputs)
                true_values.append(target)                
            val_accuracy = self.get_accuracy(
                torch.cat(logit_values, dim=0),
                torch.cat(true_values, dim=0)
            )
            if wandb.run is not None:
                wandb.log({
                    'Val/True': torch.cat(true_values, dim=0),
                    'Val/Pred': self.get_predictions(torch.cat(logit_values, dim=0)),
                })
            val_loss = running_val_loss / len(val_loader)
        return val_loss, val_accuracy

    def eval(self, data_loader=None, name='Test'):
        model = self.model
        criterion = self.criterion
        test_loader = self.test_loader if data_loader is None else data_loader
        model.eval()
        running_test_loss = 0.0
        logit_values = []
        true_values = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {key: batch[key].to(self.device) for key in batch if key != 'noise'}
                labels = batch['noise'].float().to(self.device)
                # labels = torch.round(torch.clamp(labels, 0, self.config['model']['num_bins']-1)).long().squeeze()
                target = convert_to_bins(labels, self.label_range, self.num_bins, self.device).squeeze()
                outputs = model(inputs)
                test_loss = criterion(outputs, target)
                running_test_loss += test_loss.item()
                logit_values.append(outputs)
                true_values.append(target)
            test_accuracy = self.get_accuracy(
                torch.cat(logit_values, dim=0),
                torch.cat(true_values, dim=0)
            )
            if wandb.run is not None:
                wandb.log({
                    f'{name}/True': torch.cat(true_values, dim=0),
                    f'{name}/Pred': self.get_predictions(torch.cat(logit_values, dim=0)),
                })
        test_loss = running_test_loss / len(test_loader)
        print('Finished evaluation')
        return test_loss, test_accuracy 


class TrainerMSE(Trainer):
    def __init__(self, config, model_class, optimizer_class=optim.AdamW, criterion=nn.MSELoss(), scheduler_class=optim.lr_scheduler.StepLR, epsilon=10**(-2)):
        assert config['model']['use_regression'] == True, "Model must be a regression model"
        self.epsilon = epsilon
        super().__init__(config, model_class, optimizer_class, criterion, scheduler_class)

    def get_bulk_eps_acc(self, normalized_true_arr, normalized_pred_arr):
        eps_acc = {}
        for eps in self.eps_checklist:
            acc = ((np.abs(normalized_true_arr - normalized_pred_arr)) < eps).sum() / 7500.0 * 100
            eps_acc[eps] = acc.item() 
        return eps_acc

    def get_r2_score(self, predicted, labels):
        """Return the coefficient of determination of the prediction.
        - The best possible score is 1.0 and it can be negative 
            (because the model can be arbitrarily worse). 
        - A constant model that always predicts the expected value of y, 
            disregarding the input features, would get a score of 0.0.
        """
        u = ((predicted - labels)**2).sum().item()
        v = ((labels - labels.mean())**2).sum().item()
        return 1 - u/v

    def get_accuracy(self, predicted, labels, epsilon=1/128): # 0.001):
        """
        Epsilon thresholded on accuracy 
        EPSILON can be 0 to 1.0
        epsilon 0 is same as ce.
        The area under the curve should allow to compare different models.
        """
        correct = (torch.abs(predicted - labels) <= epsilon).sum().item()
        total = labels.size(0)
        return correct / total
            
    def plot_labels_pred_wrt_distance(self, dataloader, step):
        model = self.model
        device = self.device

        true, pred, dist = [], [], []
        model.eval()
        model = model.to(device)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            dist.append(batch['direction_distance'].cpu().detach().numpy()[..., 1])
            true.append(batch['noise'].cpu().detach().numpy())
            pred.append(out.cpu().detach().numpy())
            
        dist_arr = np.concatenate(dist)
        true_arr = np.concatenate(true)
        pred_arr = np.concatenate(pred)

        pred_arr = np.clip(pred_arr, a_min=0., a_max=1.)
        fig = plt.figure(figsize=(5, 4))
        # plt.yscale('log')
        plt.scatter(dist_arr, true_arr/128., s=1.0, marker='x', label='true')
        plt.scatter(dist_arr, pred_arr, s=1.0, marker='o', c='r', label='pred')
        plt.xlabel('Distance (in meters)') # (1 -> 10 m)')
        plt.ylabel('Max Acoustic Noise (in dB)') # (1 -> 128 dB)')
        plt.legend()
        if wandb.run is not None:
            wandb.log({"dB by Distance": wandb.Image(fig)}, step=step)
        plt.savefig('dB_by_distance.png')
        plt.close(fig)
        return {'distances': dist_arr, 'true': true_arr/128., 'pred': pred_arr}

    def train_epoch(self, epoch):
        criterion = self.criterion
        train_loader = self.train_loader
        device = self.device

        self.model.train()
        running_train_loss = 0.0
        pred_values = []
        true_values = []
        for i, data in enumerate(train_loader):
            inputs = {key: data[key].to(device) for key in data if key != 'noise'}
            labels = data['noise'].float().to(device)
            labels = torch.clamp(labels, 0, 128) / 128.0
            # labels = torch.clamp(labels, 0, 128)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_train_loss += loss.item()

            # Log loss to wandb every mini-batch
            if i % self.config['summary_interval'] == 0:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (epoch, i, loss.item()))
                r2_score = self.get_r2_score(outputs, labels)
                print(f'r2 score: {r2_score:.4f}')
                if wandb.run is not None:
                    wandb.log({'R2 score': r2_score})
                pred_values.append(outputs)
                true_values.append(labels)
        train_accuracy = self.get_accuracy(
            torch.cat(pred_values, dim=0),
            torch.cat(true_values, dim=0), 
            epsilon=self.epsilon
        )
        # if wandb.run is not None:
        #     wandb.log({
        #         'Train/True': torch.cat(true_values, dim=0)*128,
        #         'Train/Pred': torch.cat(pred_values, dim=0)*128,
        #         'Train/R2': self.get_r2_score(
        #             torch.cat(pred_values, dim=0),
        #             torch.cat(true_values, dim=0), 
        #         ),
        #     })
        train_loss = running_train_loss/len(train_loader)
        return train_loss, train_accuracy

    def validate(self):
        model = self.model
        criterion = self.criterion
        val_loader = self.val_loader
        device = self.device
        
        model.eval()
        running_val_loss = 0.0
        pred_values = []
        true_values = []
        with torch.no_grad():
            for data in val_loader:
                inputs = {key: data[key].to(device) for key in data if key != 'noise'}
                labels = data['noise'].float().to(device)
                labels = torch.clamp(labels, 0, 128) / 128.0
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
                pred_values.append(outputs)
                true_values.append(labels)
                
            val_accuracy = self.get_accuracy(
                torch.cat(pred_values, dim=0),
                torch.cat(true_values, dim=0), 
                epsilon=self.epsilon
            )
            val_loss = running_val_loss / len(val_loader)
            if wandb.run is not None:
                wandb.log({
                    'Val/True': torch.cat(true_values, dim=0)*128,
                    'Val/Pred': torch.cat(pred_values, dim=0)*128,
                    'Val/R2': self.get_r2_score(
                        torch.cat(pred_values, dim=0),
                        torch.cat(true_values, dim=0), 
                    ),
                })
        return val_loss, val_accuracy

    def eval(self, data_loader=None, name='Test'):
        model = self.model
        criterion = self.criterion
        test_loader = self.test_loader if data_loader is None else data_loader
        device = self.device

        model.eval()
        running_test_loss = 0.0
        pred_values = []
        true_values = []

        with torch.no_grad():
            for data in test_loader:
                inputs = {key: data[key].to(device) for key in data if key != 'noise'}
                labels = data['noise'].float().to(device)
                labels = torch.clamp(labels, 0, 128) / 128.0
                outputs = model(inputs)
                test_loss = criterion(outputs, labels)
                running_test_loss += test_loss.item()
                pred_values.append(outputs)
                true_values.append(labels)
            test_accuracy = self.get_accuracy(
                torch.cat(pred_values, dim=0),
                torch.cat(true_values, dim=0), 
                epsilon=self.epsilon
            )
            if wandb.run is not None:
                wandb.log({
                    f'{name}/True': torch.cat(true_values, dim=0)*128,
                    f'{name}/Pred': torch.cat(pred_values, dim=0)*128,
                    f'{name}/R2': self.get_r2_score(
                        torch.cat(pred_values, dim=0),
                        torch.cat(true_values, dim=0), 
                    )   
                })
        test_loss = running_test_loss / len(test_loader)
        print('Finished evaluation')
        return test_loss, test_accuracy


if __name__=="__main__":
    # Config
    config = dict(
        device='cuda:1' if torch.cuda.is_available() else 'cpu',
        wandb={
            'project': 'anavi',
            'dir': '/scratch/vdj/wandb'
        },
        chkpt_dir='/data/vdj/ss/checkpoints/',
        chkpt_path='tmp.pth',
        data={
            'data_path': '/data/vdj/ss/anp_data_full-500/',
            'train_batch_size': 64,
            'eval_batch_size': 32,
        },
        model=dict(
            use_rgb=True, 
            use_depth=True, 
            mean_pool_visual=True, 
            use_regression=False, 
            num_bins=128,
            label_range=128,
        ),
        optim=dict(lr=0.01),
        logging=dict(
            chkpt_dir='/data/vdj/ss/checkpoints/', 
            chkpt_path="resume_visual_ce_trainer.pth"
        ),
        epochs=100, patience=25, summary_interval=100,
        resume_training=False,
        scheduler=dict(step_size=10, gamma=0.1)
    )
    print(config)
    import yaml
    stream = open('config_template.yaml', 'w')
    yaml.dump(config, stream)

    stream = open('config_template.yaml', 'r')
    config = yaml.safe_load(stream)

    trainer = TrainerCE(config, model_class=VisDirDis, optimizer_class=optim.AdamW, criterion=nn.CrossEntropyLoss(), scheduler_class=optim.lr_scheduler.StepLR)

    config['model']['use_regression'] = True
    trainer = TrainerMSE(config, model_class=VisDirDis, optimizer_class=optim.AdamW, criterion=nn.MSELoss(), scheduler_class=optim.lr_scheduler.StepLR)
    trainer.init_training()
