import torch
import torchvision.transforms as transforms
import numpy as np
import csv

from config import Config


def load_classprobs_from_csv(csv_path):
    dict_classprobs = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        filename_col = fieldnames[0]
        class_cols = fieldnames[1:]
        for row in reader:
            filename = row[filename_col]
            probs = [float(row[col]) for col in class_cols]
            dict_classprobs[filename] = probs
    return dict_classprobs

def get_device():
    """Get available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def get_unnormalize_transform():
    """Get unnormalization transform for ImageNet normalization"""
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                           std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                           std=[1., 1., 1.]),
    ])
    return unnormalize


def filenames2classprobs(filenames, dict_classprobs, n_classes=None):
    """Convert filenames to class probabilities tensor"""
    if n_classes is None:
        n_classes = Config.N_CLASSES
        
    device = get_device()
    batch_size = len(filenames)
    classprobs = np.zeros((batch_size, n_classes))
    
    for idx, filename in enumerate(filenames):
        classprobs[idx, :] = dict_classprobs[filename]
        
    return torch.from_numpy(classprobs).requires_grad_(False).type(torch.float).to(device, non_blocking=True)


def lambdas_on_the_fly(unnormalized_img, padding_size=None):
    """Compute lambda weights for patches on the fly"""
    if padding_size is None:
        padding_size = Config.PADDING_SIZE
    
    # Remove padding
    sliced_img = unnormalized_img[:, padding_size:-padding_size, padding_size:-padding_size]

    # Extract patches
    patches_unnormalized = sliced_img.unfold(1, Config.STRIDE_H, Config.STRIDE_H).unfold(2, Config.STRIDE_W, Config.STRIDE_W).reshape(3, -1, Config.STRIDE_H, Config.STRIDE_W).permute(1,0,2,3)
    
    # Convert to grayscale
    patches_unnormalized_gray = torch.mean(patches_unnormalized, 1)
    
    # Count total tissue pixels
    tissue_pixels = torch.sum((1.0 - patches_unnormalized_gray) > 1e-6)
    
    n_patches = patches_unnormalized.shape[0]
    lambdas_img = torch.zeros(n_patches, 1).type(torch.float)
    
    # Calculate lambda for each patch
    for k in range(n_patches):
        patch_k = patches_unnormalized[k]
        patch_k_gray = patches_unnormalized_gray[k]
        tissue_pixels_patch = torch.sum((1.0 - patch_k_gray) > 1e-6)
        lambdas_img[k] = tissue_pixels_patch / tissue_pixels
        
    return lambdas_img