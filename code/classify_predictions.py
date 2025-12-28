from data_utils import NiftiLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pickle
import os
from math import pi
from scipy.ndimage import label as nd_label, binary_erosion
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
from VAE_model import ResidualVAE_Segmenter

# Classify PREDICTED segmentations from decoded latent vectors
# instead of ground truth segmentations

# Load latent vectors
print("Loading latent vectors...")
with open('latent_vectors.pkl', 'rb') as f:
    encoded_data = pickle.load(f)
print(f"Loaded {len(encoded_data)} latent vectors")

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables for model path and latent dimension
MODEL_PATH = 'vae_segmenter_best_20251028-182426_epoch152_dice0.3601.pth'
LATENT_DIM = encoded_data[0]['latent_vector'].shape[0]

# Load model once on main process (will be used without multiprocessing for speed)
print("Loading VAE model...")
vae_model = ResidualVAE_Segmenter(latent_dim=LATENT_DIM, use_mean=True)
vae_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
vae_model = vae_model.to(device)
vae_model.eval()
print(f"VAE model loaded with latent_dim={LATENT_DIM} on {device}")

# Group dictionaries (kept for reference)
binary_groups = {0: "No Tumor", 1: "Tumor Present"}
contrast_enhancement_groups = {0: "No Contrast Enhancement", 1: "Low Contrast Enhancement", 2: "Medium Contrast Enhancement", 3: "High Contrast Enhancement"}
necrosis_groups = {0: "No Necrosis", 1: "Low Necrosis", 2: "Medium Necrosis", 3: "High Necrosis"}
edema_groups = {0: "No Edema", 1: "Low Edema", 2: "Medium Edema", 3: "High Edema"}
size_groups = {0: "No Tumor", 1: "Small Tumor", 2: "Medium Tumor", 3: "Large Tumor"}
positional_groups = {0: "No Tumor", 1: "Right-Up", 2: "Right-Down", 3: "Left-Up", 4: "Left-Down"}
shape_groups = {0: "No Tumor", 1: "Compact", 2: "Intermediate", 3: "Diffuse / Irregular"}
laterality_groups = {0: "No Tumor", 1: "Left Hemisphere", 2: "Right Hemisphere", 3: "Bilateral / Crossing Midline"}

# ---- Helper functions optimized for numpy 2D slices ----

def compute_compactness(mask):
    """Compute compactness for a binary mask (H,W).
    Use the largest connected component. Compactness = (perimeter**2) / (4*pi*area).
    Returns 0.0 for empty mask.
    """
    if mask.sum() == 0:
        return 0.0
    # Label connected components (background=0)
    labeled, num_feat = nd_label(mask.astype(np.uint8))
    if num_feat == 0:
        return 0.0
    # compute area per label using bincount (index 0 is background)
    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        return 0.0
    # select largest component (ignore background)
    largest_label = int(np.argmax(counts[1:]) + 1)
    area = int(counts[largest_label])
    if area <= 0:
        return 0.0
    # approximate perimeter via morphological erosion boundary
    inner = binary_erosion(labeled == largest_label)
    boundary = (labeled == largest_label) & ~inner
    perimeter = float(boundary.sum())
    return (perimeter ** 2) / (4.0 * pi * area)


def find_positional_group(seg_slice):

    tumor_mask = (seg_slice > 0) # this contains
    if tumor_mask.sum() == 0:
        return 0
    coords = np.argwhere(tumor_mask)
    cy, cx = coords.mean(axis=0)
    H, W = seg_slice.shape
    if cy < H / 2 and cx < W / 2:
        #up - left
        return 3
    elif cy < H / 2 and cx >= W / 2:
        #up - right
        return 1
    elif cy >= H / 2 and cx < W / 2:
        #down - left
        return 4
    else:
        return 2


def find_size_group_from_count(tumor_size, small_th, med_th):
    if tumor_size == 0:
        return 0
    if tumor_size <= small_th:
        return 1
    if tumor_size <= med_th:
        return 2
    return 3


def find_ratio_group_from_counts(a, b, low_th, high_th):
    if a == 0 and b == 0:
        return 0
    if b == 0:
        return 3
    ratio = a / b
    if ratio < low_th:
        return 1
    if ratio > high_th:
        return 3
    return 2


# ---- Batch processing function for decoding latent vectors ----
def decode_latent_batch(latent_vectors, vae_model, batch_size=64):
    """
    Decode a batch of latent vectors efficiently.

    Args:
        latent_vectors: List or array of latent vectors
        vae_model: VAE model
        batch_size: Number of samples to process at once

    Returns:
        List of predicted segmentation arrays
    """
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(latent_vectors), batch_size):
            batch = latent_vectors[i:i+batch_size]

            # Stack into batch tensor
            batch_tensor = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)

            # Decode batch
            seg_output, _ = vae_model.decode(batch_tensor, output_size=(240, 240))
            pred_segs = torch.argmax(seg_output, dim=1).cpu().numpy()

            all_predictions.extend(pred_segs)

    return all_predictions


# ---- Sequential processing (faster than multiprocessing for this case) ----
def process_all_samples_pass1(encoded_data, predictions):
    """Process all samples for statistics collection."""
    all_sizes = []
    all_ratios_1_2 = []
    all_ratios_1_3 = []
    all_ratios_2_3 = []
    all_enh_frac = []
    all_nec_frac = []
    all_edema_frac = []
    all_compactness = []

    for pred_seg in tqdm(predictions, desc="Computing statistics"):
        c1 = int((pred_seg == 1).sum())
        c2 = int((pred_seg == 2).sum())
        c3 = int((pred_seg == 3).sum())
        tumor_size = c1 + c2 + c3

        if tumor_size > 0:
            all_sizes.append(tumor_size)
        if c2 > 0 and c1 > 0:
            all_ratios_1_2.append(c1 / c2)
        if c3 > 0 and c1 > 0:
            all_ratios_1_3.append(c1 / c3)
        if c3 > 0 and c2 > 0:
            all_ratios_2_3.append(c2 / c3)
        if tumor_size > 0:
            if c3 > 0:
                all_enh_frac.append(c3 / tumor_size)
            if c1 > 0:
                all_nec_frac.append(c1 / tumor_size)
            if c2 > 0:
                all_edema_frac.append(c2 / tumor_size)
            all_compactness.append(compute_compactness(pred_seg > 0))

    return {
        'sizes': all_sizes,
        'ratios_1_2': all_ratios_1_2,
        'ratios_1_3': all_ratios_1_3,
        'ratios_2_3': all_ratios_2_3,
        'enh_frac': all_enh_frac,
        'nec_frac': all_nec_frac,
        'edema_frac': all_edema_frac,
        'compactness': all_compactness
    }


def process_all_samples_pass2(encoded_data, predictions, thresholds):
    """Process all samples for labeling."""
    # Unpack thresholds
    small_size_th, med_size_th = thresholds['size']
    small_r12_th, med_r12_th = thresholds['r12']
    small_r13_th, med_r13_th = thresholds['r13']
    small_r23_th, med_r23_th = thresholds['r23']
    small_enh_th, med_enh_th = thresholds['enh']
    small_nec_th, med_nec_th = thresholds['nec']
    small_edema_th, med_edema_th = thresholds['edema']
    small_comp_th, med_comp_th = thresholds['comp']

    all_labels = {
        'Positional': [],
        'Size': [],
        'ratio_1_2': [],
        'ratio_1_3': [],
        'ratio_2_3': [],
        'Binary': [],
        'Enhancement': [],
        'Necrosis': [],
        'Edema': [],
        'Shape': [],
        'Laterality': []
    }

    for idx, pred_seg in enumerate(tqdm(predictions, desc="Labeling samples")):
        case_id = encoded_data[idx]['case_id']
        slice_idx = encoded_data[idx]['slice_idx']

        c1 = int((pred_seg == 1).sum())
        c2 = int((pred_seg == 2).sum())
        c3 = int((pred_seg == 3).sum())
        tumor_size = c1 + c2 + c3

        # Compute all labels
        pos = find_positional_group(pred_seg)
        size = find_size_group_from_count(tumor_size, small_size_th, med_size_th)
        r12 = find_ratio_group_from_counts(c1, c2, small_r12_th, med_r12_th)
        r13 = find_ratio_group_from_counts(c1, c3, small_r13_th, med_r13_th)
        r23 = find_ratio_group_from_counts(c2, c3, small_r23_th, med_r23_th)
        binary = 1 if tumor_size > 0 else 0

        # Fractions grouped
        enh_frac = (c3 / tumor_size) if tumor_size > 0 and c3 > 0 else 0.0
        nec_frac = (c1 / tumor_size) if tumor_size > 0 and c1 > 0 else 0.0
        ed_frac = (c2 / tumor_size) if tumor_size > 0 and c2 > 0 else 0.0
        enh_lbl = 0 if enh_frac == 0 else (1 if enh_frac <= small_enh_th else (2 if enh_frac <= med_enh_th else 3))
        nec_lbl = 0 if nec_frac == 0 else (1 if nec_frac <= small_nec_th else (2 if nec_frac <= med_nec_th else 3))
        ed_lbl = 0 if ed_frac == 0 else (1 if ed_frac <= small_edema_th else (2 if ed_frac <= med_edema_th else 3))

        # Compactness -> shape label
        compact = compute_compactness(pred_seg > 0) if tumor_size > 0 else 0.0
        if tumor_size == 0:
            shape_lbl = 0
        elif compact <= small_comp_th:
            shape_lbl = 1
        elif compact <= med_comp_th:
            shape_lbl = 2
        else:
            shape_lbl = 3

        # Laterality
        H, W = pred_seg.shape# changed to H as the brain is always horizontal

        right_count = int(((pred_seg > 0)[:H//2, :]).sum())  # Upper half: rows 0 to H/2
        left_count = int(((pred_seg > 0)[H//2:, :]).sum())  # Lower half: rows H/2 to H
        if tumor_size == 0:
            lat = 0
        elif left_count > 0 and right_count == 0:
            lat = 1
        elif right_count > 0 and left_count == 0:
            lat = 2
        else:
            lat = 3

        all_labels['Positional'].append((case_id, slice_idx, int(pos)))
        all_labels['Size'].append((case_id, slice_idx, int(size)))
        all_labels['ratio_1_2'].append((case_id, slice_idx, int(r12)))
        all_labels['ratio_1_3'].append((case_id, slice_idx, int(r13)))
        all_labels['ratio_2_3'].append((case_id, slice_idx, int(r23)))
        all_labels['Binary'].append((case_id, slice_idx, int(binary)))
        all_labels['Enhancement'].append((case_id, slice_idx, int(enh_lbl)))
        all_labels['Necrosis'].append((case_id, slice_idx, int(nec_lbl)))
        all_labels['Edema'].append((case_id, slice_idx, int(ed_lbl)))
        all_labels['Shape'].append((case_id, slice_idx, int(shape_lbl)))
        all_labels['Laterality'].append((case_id, slice_idx, int(lat)))

    return all_labels


# ---- Decode all latent vectors in batches ----
print("\nDecoding all latent vectors...")
all_latent_vectors = [data['latent_vector'] for data in encoded_data]
predictions = decode_latent_batch(all_latent_vectors, vae_model, batch_size=64)
print(f"Decoded {len(predictions)} segmentations")

# ---- PASS 1: Statistics collection from predictions ----
print("\nPASS 1 — collecting statistics from predictions...")
stats = process_all_samples_pass1(encoded_data, predictions)

all_sizes = stats['sizes']
all_ratios_1_2 = stats['ratios_1_2']
all_ratios_1_3 = stats['ratios_1_3']
all_ratios_2_3 = stats['ratios_2_3']
all_enh_frac = stats['enh_frac']
all_nec_frac = stats['nec_frac']
all_edema_frac = stats['edema_frac']
all_compactness = stats['compactness']

print(f"Collected stats for {len(predictions)} samples")


# compute tertile thresholds (safe guards for empty lists)
def tertile(arr):
    if len(arr) == 0:
        return 0.0, 0.0
    a = np.array(arr)
    return float(np.percentile(a, 33.3333)), float(np.percentile(a, 66.6667))

small_size_th, med_size_th = tertile([s for s in all_sizes if s > 0])
small_r12_th, med_r12_th = tertile(all_ratios_1_2)
small_r13_th, med_r13_th = tertile(all_ratios_1_3)
small_r23_th, med_r23_th = tertile(all_ratios_2_3)

small_enh_th, med_enh_th = tertile(all_enh_frac)
small_nec_th, med_nec_th = tertile(all_nec_frac)
small_edema_th, med_edema_th = tertile(all_edema_frac)
small_comp_th, med_comp_th = tertile(all_compactness)

print("\nComputed thresholds:")
print(f" Size tertiles (small,med): {small_size_th:.1f}, {med_size_th:.1f}")
print(f" Ratio1/2 tertiles: {small_r12_th:.3f}, {med_r12_th:.3f}")
print(f" Ratio1/3 tertiles: {small_r13_th:.3f}, {med_r13_th:.3f}")
print(f" Ratio2/3 tertiles: {small_r23_th:.3f}, {med_r23_th:.3f}")
print(f" Enhancing fraction tertiles: {small_enh_th:.3f}, {med_enh_th:.3f}")
print(f" Necrosis fraction tertiles: {small_nec_th:.3f}, {med_nec_th:.3f}")
print(f" Edema fraction tertiles: {small_edema_th:.3f}, {med_edema_th:.3f}")
print(f" Compactness tertiles: {small_comp_th:.3f}, {med_comp_th:.3f}")

# ---- PASS 2: Labeling predictions ----
print("\nPASS 2 — labeling predictions...")

# Package thresholds
thresholds = {
    'size': (small_size_th, med_size_th),
    'r12': (small_r12_th, med_r12_th),
    'r13': (small_r13_th, med_r13_th),
    'r23': (small_r23_th, med_r23_th),
    'enh': (small_enh_th, med_enh_th),
    'nec': (small_nec_th, med_nec_th),
    'edema': (small_edema_th, med_edema_th),
    'comp': (small_comp_th, med_comp_th)
}

# Process all samples sequentially (fast with batch decoding already done)
all_labels = process_all_samples_pass2(encoded_data, predictions, thresholds)

# Unpack results
Positional_Labels = all_labels['Positional']
Size_Labels = all_labels['Size']
ratio_1_2_Labels = all_labels['ratio_1_2']
ratio_1_3_Labels = all_labels['ratio_1_3']
ratio_2_3_Labels = all_labels['ratio_2_3']
Binary_Labels = all_labels['Binary']
Enhancement_Labels = all_labels['Enhancement']
Necrosis_Labels = all_labels['Necrosis']
Edema_Labels = all_labels['Edema']
Shape_Labels = all_labels['Shape']
Laterality_Labels = all_labels['Laterality']

# Save results
outdir = "labels_out_predictions"
if not os.path.exists(outdir):
    os.makedirs(outdir)

mappings = {
    "Positional_Labels.pkl": Positional_Labels,
    "Size_Labels.pkl": Size_Labels,
    "ratio_1_2_Labels.pkl": ratio_1_2_Labels,
    "ratio_1_3_Labels.pkl": ratio_1_3_Labels,
    "ratio_2_3_Labels.pkl": ratio_2_3_Labels,
    "Binary_Labels.pkl": Binary_Labels,
    "Enhancement_Labels.pkl": Enhancement_Labels,
    "Necrosis_Labels.pkl": Necrosis_Labels,
    "Edema_Labels.pkl": Edema_Labels,
    "Shape_Labels.pkl": Shape_Labels,
    "Laterality_Labels.pkl": Laterality_Labels,
}

print("\nSaving pickles...")
for fname, data in mappings.items():
    path = os.path.join(outdir, fname)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {path} ({len(data)} entries)")