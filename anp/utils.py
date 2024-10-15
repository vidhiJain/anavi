import numpy as np
import torch 
from PIL import Image

from anp.data import (
    AudioDecibelDatasetv3, 
    AudioDecibelDatasetv2, 
    EgoViewDecibelDatasetv3,
    EgoViewDecibelDataset, 
    make_data_config, 
    make_consolidated_data_config_v3
)


def make_dataloaders(data_path, train_batch_size, eval_batch_size, load_images=True, data_class='pano', num_workers=8, pin_memory=True, prefetch_factor=2, shard_folders=[], **kwargs):
    if data_class == 'ego':
        train_dataset = EgoViewDecibelDataset(make_data_config('train', data_path, load_images))
        val_dataset = EgoViewDecibelDataset(make_data_config('val', data_path, load_images))
        test_dataset = EgoViewDecibelDataset(make_data_config('test', data_path, load_images))
    elif data_class == 'pano':
        train_dataset = AudioDecibelDatasetv2(make_data_config('train', data_path, load_images))
        val_dataset = AudioDecibelDatasetv2(make_data_config('val', data_path, load_images))
        test_dataset = AudioDecibelDatasetv2(make_data_config('test', data_path, load_images))
    elif data_class == 'shard_dirdis':
        config = make_consolidated_data_config_v3(shard_folders, split='train', load_images=False, **kwargs)
        train_dataset = AudioDecibelDatasetv3(config)
        # keep the same val and test splits
        val_dataset = AudioDecibelDatasetv2(make_data_config('val', data_path, load_images, **kwargs))
        test_dataset = AudioDecibelDatasetv2(make_data_config('test', data_path, load_images, **kwargs))
    elif data_class == 'shard_pano':
        config = make_consolidated_data_config_v3(shard_folders, split='train', load_images=True, **kwargs)
        train_dataset = AudioDecibelDatasetv3(config)
        # keep the same val and test splits
        val_dataset = AudioDecibelDatasetv2(make_data_config('val', data_path, load_images, **kwargs))
        test_dataset = AudioDecibelDatasetv2(make_data_config('test', data_path, load_images, **kwargs))
    elif data_class == 'shard_ego':
        config = make_consolidated_data_config_v3(shard_folders, split='train', load_images=True, **kwargs)
        train_dataset = EgoViewDecibelDatasetv3(config)
        # keep the same val and test splits
        val_dataset = EgoViewDecibelDataset(make_data_config('val', data_path, load_images, **kwargs))
        test_dataset = EgoViewDecibelDataset(make_data_config('test', data_path, load_images, **kwargs))
    else:
        raise ValueError(f"Data class {data_class} not recognized.")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    return train_loader, val_loader, test_loader


def get_accuracy(outputs, labels):
    # Get the predicted labels by taking the argmax of the logits along the class dimension
    _, predicted = torch.max(outputs, 1)
    # Compare the predicted labels with the true labels
    correct = (predicted == labels).sum().item()
    # Compute the accuracy as the ratio of correct predictions to total predictions
    accuracy = correct / labels.size(0)
    return accuracy


def get_accuracy_eps(outputs, labels, epsilon=2): 
    """
    Epsilon thresholded on accuracy 
    EPSILON can be 0, 2, 4, 8, 16, 32, 64, 128
    The area under the curve should allow to compare different models.
    """
    _, predicted = torch.max(outputs, 1)
    correct = (torch.abs(predicted - labels) <= epsilon).sum().item()
    total = labels.size(0)
    return correct / total

def convert_to_bins(numbers, n, m, device):
    """
    Converts a list of numbers from 0 to n into m bins using PyTorch.
    
    Parameters:
    numbers (torch.Tensor): Tensor of numbers to be binned.
    n (float): The maximum value in the range [0, n].
    m (int): The number of bins.
    
    Returns:
    torch.Tensor: The bin indices corresponding to each number.
    """
    # Ensure numbers are within the range [0, n]
    numbers = torch.clamp(numbers, 0, n)
    
    # Determine bin edges
    bin_edges = torch.linspace(0, n, steps=m+1).to(device)
    
    # Digitize the numbers into bins
    bin_indices = torch.bucketize(numbers, bin_edges, right=True) - 1  # -1 to make bins 0-indexed
    
    # Handle edge case where a number equals n
    bin_indices[bin_indices == m] = m - 1
    
    return bin_indices


def bin_to_centroid(bin_index, n, m):
    """
    Converts a bin index back to the centroid of the bin using PyTorch.
    
    Parameters:
    bin_index (torch.Tensor): The index of the bin.
    n (float): The maximum value in the range [0, n].
    m (int): The number of bins.
    
    Returns:
    torch.Tensor: The centroid value of the bin.
    """
    # Determine bin width
    bin_width = n / m
    
    # Calculate the centroid of the bin
    centroids = (bin_index + 0.5) * bin_width
    
    return centroids
