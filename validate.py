#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from data_preprocessing import get_transforms, ImageFolderWithFilenames
from utils import *
from model import create_swin_transformer

def load_and_evaluate(checkpoint_path, testloader, dict_classprobs, device, class_names, reshape_transform):
    net = create_swin_transformer().to(device)
    net.load_state_dict(torch.load(checkpoint_path), strict=False)
    net.eval()
    
    unnormalize = get_unnormalize_transform()
    nb_classes = len(class_names)
    cm = np.zeros((nb_classes, nb_classes), dtype=int)
    
    with torch.no_grad():
        for inputs, classes, filenames in testloader:
            inputs = inputs.to(device)
            unnorm = unnormalize(torch.squeeze(inputs))
            patches = torch.squeeze(inputs)\
                .unfold(1, Config.PATCH_H, Config.STRIDE_H)\
                .unfold(2, Config.PATCH_W, Config.STRIDE_W)\
                .reshape(3, -1, Config.PATCH_H, Config.PATCH_W)\
                .permute(1,0,2,3)
            patches = reshape_transform(patches)
            lambdas = lambdas_on_the_fly(unnorm).repeat(1,3).to(device)
            out_patches = F.softmax(net(patches), dim=1)
            probs = torch.sum(lambdas * out_patches, dim=0)
            pred = np.argmax(probs.cpu().numpy())
            true = classes.item()
            cm[true, pred] += 1
    
    return cm

def save_confusion_matrix(cm, class_names, output_path, title):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names).astype(int)
    df_cm_n = pd.DataFrame(cm_norm, index=class_names, columns=class_names).astype(float)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=df_cm, annot_kws={'va':'bottom', "size": 12}, 
                fmt="", cbar=False, cmap='Blues')
    sns.heatmap(df_cm_n, annot=df_cm_n, annot_kws={'va':'top', "size": 12}, 
                fmt='.2%', cbar=True, cmap='Blues', vmin=0.0, vmax=1.0)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    dict_cp = load_classprobs_from_csv(args.csv)
    
    _, test_t, _, reshape_t = get_transforms()
    test_dataset = ImageFolderWithFilenames(
        root=os.path.join(Config.WORKING_DIR, "test_ABD"),
        transform=test_t
    )
    from torch.utils.data import DataLoader
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_names = Config.CLASS_NAMES[:Config.N_CLASSES]
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_cms = []
    
    # Process each fold
    for fold in range(1, args.num_folds + 1):
        checkpoint_path = args.checkpoint_pattern.format(fold)
        print(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            continue
            
        cm = load_and_evaluate(checkpoint_path, testloader, dict_cp, device, class_names, reshape_t)
        all_cms.append(cm)
        
        # Save individual fold
        np.save(os.path.join(args.output_dir, f"cm_fold{fold}.npy"), cm)
        save_confusion_matrix(cm, class_names, 
                            os.path.join(args.output_dir, f"cm_fold{fold}.png"), 
                            f"Fold {fold}")
    
    # Save combined
    if all_cms:
        combined_cm = np.sum(all_cms, axis=0)
        np.save(os.path.join(args.output_dir, "cm_combined.npy"), combined_cm)
        save_confusion_matrix(combined_cm, class_names,
                            os.path.join(args.output_dir, "cm_combined.png"),
                            "Combined")

if __name__ == "__main__":
    main()