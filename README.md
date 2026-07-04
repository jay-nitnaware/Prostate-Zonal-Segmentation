# Attention U-Net with HRNet-W32 Encoder — Prostate Zonal Segmentation

A PyTorch pipeline for automatic segmentation of prostate zonal anatomy from T2-weighted MRI. The model combines an **HRNet-W32** backbone (via `timm`) with an **Attention U-Net** style decoder, trained with a combined Dice + Focal loss and evaluated using patient-level 5-fold cross-validation.

## Overview

- **Task:** Multi-class semantic segmentation of prostate zones on 2D MRI slices
- **Architecture:** HRNet-W32 encoder (feature-extractor mode) + attention-gated U-Net decoder
- **Classes (5):**

  | Label | Zone | Description |
  |---|---|---|
  | 0 | Background | Everything outside the prostate zones |
  | 1 | AFMS | Anterior Fibromuscular Stroma |
  | 2 | PZ | Peripheral Zone |
  | 3 | TZ | Transition Zone |
  | 4 | ProstaticUrethra | Prostatic urethra |

- **Validation strategy:** Patient-level 80/20 train+val / test split, followed by 5-fold cross-validation on the train+val patients
- **Best fold's model** is copied out as `best_model.pth` and used for final test-set evaluation

## Dataset

- **Source:** [ProstateX Zones Segmentations](https://www.cancerimagingarchive.net/analysis-result/prostatex-seg-zones/) on The Cancer Imaging Archive (TCIA)
- 98 patients, all 4 foreground zones present for every patient, no voxel overlap between zones (verified in the notebook)
- **MRI volumes:** `mri/ProstateX-XXXX/ProstateX-XXXX_0000.nii.gz` — shape `(384, 384, N)`, `int16`
- **Masks:** one binary NIfTI file per zone per patient:
  - `masks/ProstateX-XXXX/AFMS.nii.gz`
  - `masks/ProstateX-XXXX/PZ.nii.gz`
  - `masks/ProstateX-XXXX/TZ.nii.gz`
  - `masks/ProstateX-XXXX/ProstaticUrethra.nii.gz`
- Each mask file stores its own native nonzero label; the loader thresholds every file at `> 0` and remaps it to the project's unified class scheme via `ZONE_MAP`.

The notebook expects the data to already be organized in this `mri/` + `masks/` folder layout (e.g. converted from the ProstateX dataset). Update `MRI_ROOT` and `SEG_ROOT` at the top of the notebook to point to your local copy.

## Preprocessing

Implemented in `ProstateZonalDataset` (2D slice-level dataset built from the 3D volumes):

1. Load each MRI volume and apply per-volume **z-score normalization**.
2. Build a single-channel integer mask by writing each zone's thresholded region with its mapped class ID.
3. Extract 2D axial slices from the (H, W, S) volume — one dataset item per slice.
4. Resize image (bilinear) and mask (nearest-neighbor) to `TARGET_SIZE = (320, 320)`.
5. Return `(image, mask, patient_id, slice_index)` so predictions can be traced back to a specific patient/slice for evaluation and visualization.

Patients are split at the **patient level** (not slice level) to prevent data leakage between train, validation, and test sets.

## Model Architecture

`AttentionUNetHRNetW32`:

- **Encoder:** `timm.create_model('hrnet_w32', features_only=True)`, with the stem `conv1` replaced to accept single-channel (grayscale MRI) input instead of 3-channel RGB. Feature channel counts are extracted dynamically via a dummy forward pass.
- **Decoder:** Four `DecoderBlock` stages, each performing:
  - Transposed-convolution upsampling
  - An `AttentionGate` that gates encoder skip-connection features using the decoder's upsampled signal (Attention U-Net mechanism)
  - Two Conv-BN-ReLU blocks after concatenating the gated skip connection
- **Output head:** A final upsampling block plus a `1x1` convolution producing per-pixel logits over the 5 classes, with a safety interpolation step to guarantee the output matches the input resolution exactly.

## Loss Function

A combined **Dice + Focal** loss (`DiceFocalLoss`), with per-class weights (`CLASS_WEIGHTS`) to counteract class imbalance (background dominates the image; the urethra and AFMS are small structures):

- `DiceLoss`: soft Dice computed per class, weighted and averaged
- `FocalLoss`: focal loss (α = 0.25, γ = 2.0) on top of weighted cross-entropy to focus training on hard/minority pixels
- Final loss = Dice + Focal

Per-class Dice is also tracked as a training/validation metric via `get_batch_dice`.

## Training

Configuration (defined near the top of the notebook):

| Parameter | Value |
|---|---|
| Target image size | 320 × 320 |
| Batch size | 2 |
| Epochs (max) | 50 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Cross-validation folds | 5 |
| Test split | 20% of patients |
| Early stopping patience | 10 epochs (on validation loss) |
| Seed | 42 |

Training flow (`run_kfold` → `train_one_fold`):

1. Split the train+val patients into 5 folds with `KFold` (patient-level, shuffled).
2. For each fold: train `AttentionUNetHRNetW32` for up to 50 epochs, tracking train/val loss and per-class Dice each epoch.
3. Save the best checkpoint per fold (lowest val loss), periodic checkpoints every 5 epochs, and a per-fold metrics JSON.
4. Apply early stopping if validation loss doesn't improve for 10 consecutive epochs.
5. After all folds, aggregate mean ± std Dice per class across folds, save a `*_kfold_summary.json`, and copy the globally best-performing fold's weights to `best_model.pth`.

> **Note:** The notebook was authored for a Windows/local GPU environment (`num_workers=0` in `DataLoader`s to avoid multiprocessing issues, and the k-fold training loop is wrapped in `if __name__ == '__main__':`). Update `MRI_ROOT` / `SEG_ROOT` paths and `DEVICE` handling as needed for your environment.

## Evaluation

Once `best_model.pth` is available:

- **`evaluate_on_test`** — runs inference over the held-out test patients, computes per-class and mean foreground Dice, and saves overlay visualizations (MRI / ground truth / prediction) for a random sample of test patients (all slices for each sampled patient).
- **`plot_confusion_matrix`** — computes a row-normalized (recall) pixel-level confusion matrix across all 5 classes on the test set and saves it as a heatmap.
- Aggregate result cells compute and print per-fold validation Dice (from saved metrics JSON files) alongside final test-set Dice per zone.

## Outputs

All artifacts are written under `AttentionUNet_HRNetW32/`:

```
AttentionUNet_HRNetW32/
├── checkpoints/
│   ├── attn_unet_hrnetw32_fold{1..5}_best.pth
│   ├── attn_unet_hrnetw32_fold{1..5}_epoch{N}.pth
│   ├── attn_unet_hrnetw32_fold{1..5}_metrics.json
│   └── attn_unet_hrnetw32_kfold_summary.json
├── plots/
│   ├── sample_slice.png
│   ├── all_folds_loss.png
│   ├── confusion_matrix.png
│   └── (per-fold training curves)
├── visuals/
│   └── AttnUNet-HRNetW32/
│       └── {patient_id}_slice{NNN}.png   # MRI / GT / prediction overlays
├── best_model.pth
└── test_metrics.json
```

## Requirements

- Python 3.x
- PyTorch (with CUDA support recommended)
- `timm` (HRNet-W32 backbone)
- `nibabel` (NIfTI I/O)
- `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

Install with:

```bash
pip install torch torchvision timm nibabel numpy scikit-learn matplotlib seaborn tqdm
```

## Usage

1. Organize your data as `mri/ProstateX-XXXX/ProstateX-XXXX_0000.nii.gz` and `masks/ProstateX-XXXX/{AFMS,PZ,TZ,ProstaticUrethra}.nii.gz`.
2. Update `MRI_ROOT` and `SEG_ROOT` in the configuration cell to point to your dataset.
3. Run the notebook top to bottom:
   - Cells 1–6: setup, config, dataset construction, and a sanity-check visualization
   - Cell 7: model definition and a forward-pass sanity check
   - Cells 8–10: loss functions and the 5-fold training loop (`run_kfold`)
   - Cells 11–13: reload saved fold histories/checkpoints
   - Cells 14–17: test-set evaluation, confusion matrix, and cross-validation/test summary reporting

## Notes

- Class weights (`CLASS_WEIGHTS = [0.05, 0.30, 0.20, 0.15, 0.30]`) down-weight the dominant background class and up-weight the smaller AFMS/urethra structures.
- The pipeline is slice-based (2D), not full 3D volumetric segmentation — each MRI slice is treated as an independent sample, with patient ID tracked for correct train/val/test grouping and later aggregation.
