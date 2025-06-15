# PropSegNet


A PyTorch pipeline for classifying and segmenting three distinct vascular patterns in CD31-stained tissue sections. 
This repo provides scripts for training and validation script.

---

## 🚀 Installation

1. Clone this repo  
   ```bash
   git clone https://github.com/your-username/PropSegNet.git
   cd PropSegNe

## 📥 Pre-trained Checkpoints

You can download our trained PropSegNet models from Google Drive: https://drive.google.com/drive/folders/1SKg3PC9lNUy2enrjbo_RyFPeGX9dfw89?usp=sharing

## ✅ Validation
The new validate.py script loads a saved checkpoint, runs inference on the ABD test set, computes both raw and normalized confusion matrices, and saves:

- results/y_true.npy, results/y_pred.npy

- results/cm_fold{fold}.png, results/cm_fold{fold}.npy

### Usage
python validate.py \
  --checkpoint "./checkpoints/SwinT_ABD_zeropadding_fold{}.pt" \
  --csv "multi_gt_prob.csv" \
  --output-dir results

Arguments

--checkpoint Path to .pt model file.

--csv Ground-truth probabilities CSV.

--output-dir Directory to write .npy arrays and the PNG plot.

### 📈 Results
Open results/confusion_matrix.png to visually inspect per-class performance.
Load the .npy files for custom analysis in NumPy or pandas.

