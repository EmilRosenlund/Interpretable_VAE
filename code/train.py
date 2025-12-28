import os
import torch
import torch.nn as nn
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch.nn.functional as F
import numpy as np
import nibabel as nib

from data_utils import *
from VAE_model import *



if __name__ == "__main__":

    # ---------------- Initialize model ----------------
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    NORMALIZATION_MODE = 'mustd'
    batch_size = 16
    num_workers = 15
    lr = 8e-5
    epochs = 300

    v_ce = 0.3
    v_dice = 0.8
    v_recon = 0.2
    tumor_slice_ratio = 0.65
    note = f"Using {NORMALIZATION_MODE} for preprocessing normalization, using instance norm, Using CE and extra epochs"
    print(f"NOTE: {note} ")
    print(f'Batch size: {batch_size}, Num workers: {num_workers}, Learning rate: {lr}, Epochs: {epochs}')
    print(f'V_CE: {v_ce}, V_DICE: {v_dice}, V_RECON: {v_recon}')

    # ---------------- Load dataset ----------------
    loader = NiftiLoader(folder_path="C:\\Users\\emilr\\Documents\\GitHub\\AAU_P7\\data\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData")
    #BratsDatasetClassAware(Dataset):
    """ def __init__(self, case_ids, nii_files, target_size=(128,128), tumor_slice_ratio=0.7,
                    rare_class_ratio=0.3, validate=False, val_slices_per_case=3,
                    validation_tumor_ratio=0.8, rare_classes=[1,3]): """
    dataset = BratsDatasetBalanced(loader.case_ids, loader.nii_files, tumor_slice_ratio=tumor_slice_ratio, normalization_mode = NORMALIZATION_MODE)  # pass nii_files too
    print(f'Dataset initialized with {len(dataset)} samples.')

    val_loader = NiftiLoader(folder_path="C:\\Users\\emilr\\Documents\\GitHub\\AAU_P7\\data\\BraTS2020_ValidationData")
    #Load full valdidation dataset
    val_dataset = BratsDatasetBalanced(val_loader.case_ids, val_loader.nii_files,
                                    tumor_slice_ratio=0.7, validate=True, val_slices_per_case=10, normalization_mode = NORMALIZATION_MODE)
    print(f'Validation Dataset initialized with {len(val_dataset)} slices across {len(val_loader.case_ids)} cases (balanced sampling).')

    # ---------------- Setup device ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ---------------- DataLoader ----------------
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f'Total samples in training dataset: {len(dataloader.dataset)} (random slices)')
    print(f'Total samples in validation dataset: {len(val_dataloader.dataset)} (all slices - comprehensive)')

    # ---------------- Model, optimizer, scheduler ----------------
    model = ResidualVAE_Segmenter(v_ce=v_ce, v_dice=v_dice, v_recon=v_recon).to(device)

    # ---------------- Compute class volumes for generalized dice loss ----------------
    print("=" * 60)
    print("COMPUTING DATASET CLASS VOLUMES FOR GENERALIZED DICE LOSS")
    print("=" * 60)
    model.compute_class_volumes(dataloader)
    print("=" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # bigger LR for focal+KL
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # smoother decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)  # Reduce LR when loss plateaus

    # ---------------- Training loop ----------------
    best_path, final_path = model.training_loop(dataloader, optimizer=optimizer, scheduler=scheduler, epochs=epochs, val_dataloader=val_dataloader)

    print("🎉 Training completed!")
    if best_path:
        print(f"✅ Best model saved to: {best_path}")
        print(f"📁 Final model saved to: {final_path}")
    else:
        # Fallback if something went wrong
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        fallback_path = f"vae_segmenter_{timestamp}.pth"
        model.save_model(fallback_path)
        print(f"💾 Fallback: Saved current model to {fallback_path}")

    # ---------------- Final Comprehensive Validation ----------------
    val_dataset = BratsDatasetBalanced(val_loader.case_ids, val_loader.nii_files,
                                    tumor_slice_ratio=0.7, validate=True, val_slices_per_case=-1, normalization_mode = NORMALIZATION_MODE)
    print("\n" + "="*80)
    print("STARTING FINAL COMPREHENSIVE VALIDATION ON ALL SLICES...")
    print("="*80)
    # Load the best model for comprehensive validation
    if best_path:
        print(f"Loading best model: {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))

    model.eval()
    comprehensive_loss, comprehensive_logs, dice_3D = model.validate(val_dataloader, epoch=epochs, class_groups= {"TC": (1,3), "WT": (1,2,3), "NT": (1), "ED": (2), "ET": (3) , "B": (0)})

    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE VALIDATION RESULTS - ALL SLICES")
    print("="*80)
    print(f"Dice Scores 3D {dice_3D}")
    print(f"Focal Loss (Cross-Entropy):     {comprehensive_logs['ce']:.6f}")
    print(f"Dice Loss:                      {comprehensive_logs['dice']:.6f}")
    print(f"L2 Loss (Reconstruction):       {comprehensive_logs['recon']:.6f}")
    print(f"KL Loss (VAE Regularization):   {comprehensive_logs['kl']:.6f}")
    print(f"Total Weighted Loss:            {comprehensive_loss:.6f}")
    print("="*80)
    print(f"Total slices processed:         {len(val_dataset)}")
    print(f"Validation cases:               {len(val_loader.case_ids)}")