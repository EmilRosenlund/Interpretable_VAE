"""
Encode all slices in the dataset using the VAE encoder and save latent vectors
along with case ID and slice index for later use with labels.
"""

from data_utils import NiftiLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pickle
import os
import torch
from VAE_model import ResidualVAE_Segmenter
from multiprocessing import Pool, cpu_count

# Configuration
MODEL_PATH = "vae_segmenter_best_20251017-170514_epoch150_dice0.3689.pth"
DATA_PATH = "/workspace/data2/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
OUTPUT_FILE = "latent_vectors.pkl"
LATENT_DIM = 128
BATCH_SIZE = 32  # Process multiple slices at once for efficiency
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Load the VAE model
print(f"\nLoading model from {MODEL_PATH}...")
model = ResidualVAE_Segmenter(latent_dim=LATENT_DIM, n_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# Load dataset information
print(f"\nLoading dataset from {DATA_PATH}...")
loader = NiftiLoader(folder_path=DATA_PATH)
case_ids = loader.case_ids
nii_files = loader.nii_files
print(f"Loaded {len(case_ids)} cases")

def load_case_multichannel(case_id):
    """Load all 4 modalities for a given case and return as (H, W, D, 4) array."""
    try:
        paths = nii_files[case_id]
        
        # Load each modality
        flair_vol = nib.load(paths['flair']).get_fdata()
        t1_vol = nib.load(paths['t1']).get_fdata()
        t1ce_vol = nib.load(paths['t1ce']).get_fdata()
        t2_vol = nib.load(paths['t2']).get_fdata()
        
        # Stack along channel dimension: (H, W, D, 4)
        X = np.stack([flair_vol, t1_vol, t1ce_vol, t2_vol], axis=-1)
        
        # Normalize each modality independently
        for i in range(4):
            channel = X[..., i]
            mean = channel.mean()
            std = channel.std()
            if std > 0:
                X[..., i] = (channel - mean) / (std + 1e-8)
            else:
                X[..., i] = 0.0
        
        return X
    except Exception as e:
        print(f"Error loading case {case_id}: {e}")
        return None

def encode_case(case_id):
    """
    Load a case, encode all slices, and return a list of 
    (case_id, slice_idx, latent_vector) tuples.
    """
    X = load_case_multichannel(case_id)
    if X is None:
        return []
    
    num_slices = X.shape[2]
    results = []
    
    # Process slices in batches
    with torch.no_grad():
        for start_idx in range(0, num_slices, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_slices)
            batch_slices = []
            batch_indices = []
            
            # Collect batch of slices
            for slice_idx in range(start_idx, end_idx):
                # Extract slice (H, W, 4) and transpose to (4, H, W) for PyTorch
                slice_data = X[:, :, slice_idx, :]  # (H, W, 4)
                slice_data = np.transpose(slice_data, (2, 0, 1))  # (4, H, W)
                batch_slices.append(slice_data)
                batch_indices.append(slice_idx)
            
            # Stack into batch tensor (B, 4, H, W)
            batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).float().to(DEVICE)
            
            # Encode batch
            mu, logvar = model.encode(batch_tensor)
            
            # Use mean (mu) as the latent representation (not sampling)
            latent_vectors = mu.cpu().numpy()
            
            # Store results
            for slice_idx, latent_vec in zip(batch_indices, latent_vectors):
                results.append({
                    'case_id': case_id,
                    'slice_idx': slice_idx,
                    'latent_vector': latent_vec
                })
    
    return results

# Main encoding loop
print("\n=== Encoding all slices ===")
all_encoded_data = []

for case_id in tqdm(case_ids, desc="Encoding cases"):
    encoded_slices = encode_case(case_id)
    all_encoded_data.extend(encoded_slices)

print(f"\nTotal encoded slices: {len(all_encoded_data)}")

# Save to pickle file
print(f"\nSaving encoded data to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(all_encoded_data, f)

print(f"Saved {len(all_encoded_data)} encoded slices to {OUTPUT_FILE}")

# Print some statistics
print("\n=== Statistics ===")
print(f"Total cases processed: {len(case_ids)}")
print(f"Total slices encoded: {len(all_encoded_data)}")
if len(all_encoded_data) > 0:
    print(f"Latent vector dimension: {all_encoded_data[0]['latent_vector'].shape}")
    print(f"First entry example:")
    print(f"  Case ID: {all_encoded_data[0]['case_id']}")
    print(f"  Slice index: {all_encoded_data[0]['slice_idx']}")
    print(f"  Latent vector shape: {all_encoded_data[0]['latent_vector'].shape}")

print("\n=== All done! ===")
print(f"\nYou can now use '{OUTPUT_FILE}' with the label files:")
