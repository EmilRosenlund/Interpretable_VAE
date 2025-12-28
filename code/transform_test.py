import pickle
import numpy as np
import torch
from VAE_model_up import ResidualVAE_Segmenter
from tqdm import tqdm
from math import pi
from scipy.ndimage import label as nd_label, binary_erosion

# -----------------------------
# Load latent vectors and GMMs
# -----------------------------
print("Loading data...")
with open("latent_vectors.pkl", "rb") as f:
    encoded_data = pickle.load(f)

with open("gmm_models.pkl", "rb") as f:
    gmm_models = pickle.load(f)

# Load all label files
label_files = {
    'Size': 'labels_out_predictions/Size_Labels.pkl',
    'Positional': 'labels_out_predictions/Positional_Labels.pkl',
    'Enhancement': 'labels_out_predictions/Enhancement_Labels.pkl',
    'Necrosis': 'labels_out_predictions/Necrosis_Labels.pkl',
    'Edema': 'labels_out_predictions/Edema_Labels.pkl',
    'Shape': 'labels_out_predictions/Shape_Labels.pkl',
    'Laterality': 'labels_out_predictions/Laterality_Labels.pkl'
}

all_labels = {}
for semantic, filepath in label_files.items():
    with open(filepath, "rb") as f:
        all_labels[semantic] = pickle.load(f)
    print(f"Loaded {semantic} labels")

# -----------------------------
# Load VAE model
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResidualVAE_Segmenter()
model.load_state_dict(torch.load("vae_segmenter_best_20251028-182426_epoch152_dice0.3601.pth", map_location=device))
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# -----------------------------
# Define semantic configurations
# -----------------------------
semantic_configs = {
    'Size': {
        'source_class': 1,  # Small
        'target_class': 2,  # Medium
        'critical_dims': [3, 7, 11, 34, 60, 66, 74, 76, 92, 101],
        'groups': {0: "No Tumor", 1: "Small Tumor", 2: "Medium Tumor", 3: "Large Tumor"},
        'metric': 'area'  # absolute area
    },
    'Positional': {
        'source_class': 1,  # Right-Up
        'target_class': 2,  # Right-Down
        'critical_dims': [],  # Will be populated if available
        'groups': {0: "No Tumor", 1: "Right-Up", 2: "Right-Down", 3: "Left-Up", 4: "Left-Down"},
        'metric': 'centroid'
    },
    'Enhancement': {
        'source_class': 1,  # Low
        'target_class': 2,  # Medium
        'critical_dims': [4, 5, 7, 11, 13, 20, 34, 39, 42, 43, 44, 47, 51, 59, 60, 62, 69, 70, 74, 75, 76, 86, 92, 97, 101, 107, 108, 112, 119],
        'groups': {0: "No Enhancement", 1: "Low Enhancement", 2: "Medium Enhancement", 3: "High Enhancement"},
        'metric': 'ratio',  # c3 / total
        'class_id': 3
    },
    'Necrosis': {
        'source_class': 1,  # Low
        'target_class': 2,  # Medium
        'critical_dims': [],  # Will be populated if available
        'groups': {0: "No Necrosis", 1: "Low Necrosis", 2: "Medium Necrosis", 3: "High Necrosis"},
        'metric': 'ratio',  # c1 / total
        'class_id': 1
    },
    'Edema': {
        'source_class': 1,  # Low
        'target_class': 2,  # Medium
        'critical_dims': [7, 11, 13, 39, 44, 56, 60, 70, 74, 92, 101, 107],
        'groups': {0: "No Edema", 1: "Low Edema", 2: "Medium Edema", 3: "High Edema"},
        'metric': 'ratio',  # c2 / total
        'class_id': 2
    },
    'Shape': {
        'source_class': 1,  # Compact
        'target_class': 2,  # Intermediate
        'critical_dims': [],  # Will be populated if available
        'groups': {0: "No Tumor", 1: "Compact", 2: "Intermediate", 3: "Diffuse / Irregular"},
        'metric': 'compactness'
    },
    'Laterality': {
        'source_class': 1,  # Left
        'target_class': 2,  # Right
        'critical_dims': [],  # Will be populated if available
        'groups': {0: "No Tumor", 1: "Left Hemisphere", 2: "Right Hemisphere", 3: "Bilateral"},
        'metric': 'centroid'
    }
}

# Try to load critical dimensions from saved results
try:
    with open("critical_dimensions_0.16_results.pkl", "rb") as f:
        critical_dims_results = pickle.load(f)
    print("\nLoaded critical dimensions from critical_dimensions_0.16_results.pkl")
    for semantic, results in critical_dims_results.items():
        if semantic in semantic_configs:
            semantic_configs[semantic]['critical_dims'] = results['critical_dims']
            print(f"  {semantic}: {len(results['critical_dims'])} dimensions")
except FileNotFoundError:
    print("\nCritical dimensions file not found, using hardcoded values")

# -----------------------------
# -----------------------------
# Transform function
# -----------------------------
def transform_to_gmm(latent_vector, gmm_target, alpha=1.0, critical_dims=None):
    if critical_dims is None:
        critical_dims = np.arange(latent_vector.shape[0])
    component_idx = np.argmax(gmm_target.weights_)
    target_mean = gmm_target.means_[component_idx]
    z_transformed = latent_vector.copy()
    latent_centered = latent_vector[critical_dims] - np.mean(latent_vector[critical_dims])
    z_transformed[critical_dims] = target_mean[critical_dims] + alpha * latent_centered
    return z_transformed

# -----------------------------
# Helper to decode
# -----------------------------
def decode_latent(z):
    with torch.no_grad():
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        seg_output, _ = model.decode(z_tensor, output_size=(240, 240))
        pred_seg = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
    return pred_seg

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

def compute_centroid_metrics(mask):
    """Compute centroid position in pixels.
    Returns (vertical_pos, horizontal_pos) in pixel coordinates.
    """
    if mask.sum() == 0:
        return 0.0, 0.0  # (vertical_pos, horizontal_pos)

    coords = np.argwhere(mask)
    cy, cx = coords.mean(axis=0)  # row (vertical), col (horizontal) in pixels

    return cy, cx

# -----------------------------
# Process a single semantic
# -----------------------------
def process_semantic(semantic_name, config, alpha_value=2.0):
    """Process transformations for a single semantic at given alpha value"""

    print(f"\n{'='*70}")
    print(f"Processing {semantic_name}")
    print('='*70)

    # Check if classes exist
    source_class = config['source_class']
    target_class = config['target_class']

    if source_class not in gmm_models[semantic_name] or target_class not in gmm_models[semantic_name]:
        print(f"Skipping {semantic_name}: required classes not available")
        return None

    # Get GMM for target
    H_target = gmm_models[semantic_name][target_class]['gmm']

    # Get source indices (limit to 1000 samples)
    labels = all_labels[semantic_name]
    source_indices = [idx for idx, (_, _, label) in enumerate(labels) if label == source_class]

    if len(source_indices) > 1000:
        source_indices = source_indices[:1000]

    if len(source_indices) == 0:
        print(f"No samples found for {semantic_name} class {source_class}")
        return None

    print(f"Source class: {config['groups'][source_class]} (n={len(source_indices)})")
    print(f"Target class: {config['groups'][target_class]}")
    print(f"Critical dimensions: {len(config['critical_dims'])} dims - {config['critical_dims'][:10]}...")
    print(f"Metric: {config['metric']}")

    # Initialize storage
    changes_minus = []
    changes_plus = []
    tumor_disappeared_minus = []
    tumor_disappeared_plus = []

    # Additional metrics for laterality
    if config['metric'] == 'centroid':
        vertical_shifts_minus = []
        vertical_shifts_plus = []
        horizontal_shifts_minus = []
        horizontal_shifts_plus = []

    for idx in tqdm(source_indices, desc=f"  Alpha ±{alpha_value}"):
        latent_vector = encoded_data[idx]['latent_vector']

        # Original prediction
        pred_original = decode_latent(latent_vector)
        tumor_size_original = (pred_original > 0).sum()
        tumor_mask_original = pred_original > 0

        # Calculate original value based on metric type
        if config['metric'] == 'area':
            value_original = tumor_size_original
        elif config['metric'] == 'ratio':
            class_id = config['class_id']
            class_count_original = (pred_original == class_id).sum()
            value_original = (class_count_original / tumor_size_original) if tumor_size_original > 0 else 0.0
        elif config['metric'] == 'compactness':
            value_original = compute_compactness(tumor_mask_original) if tumor_size_original > 0 else 0.0
        elif config['metric'] == 'centroid':
            vert_orig, horiz_orig = compute_centroid_metrics(tumor_mask_original)
            value_original = vert_orig  # Use vertical position as primary metric

        # Alpha negative
        z_minus = transform_to_gmm(latent_vector, H_target, alpha=-alpha_value, critical_dims=config['critical_dims'])
        pred_minus = decode_latent(z_minus)
        tumor_size_minus = (pred_minus > 0).sum()
        tumor_mask_minus = pred_minus > 0

        if config['metric'] == 'area':
            value_minus = tumor_size_minus
        elif config['metric'] == 'ratio':
            class_count_minus = (pred_minus == class_id).sum()
            value_minus = (class_count_minus / tumor_size_minus) if tumor_size_minus > 0 else 0.0
        elif config['metric'] == 'compactness':
            value_minus = compute_compactness(tumor_mask_minus) if tumor_size_minus > 0 else 0.0
        elif config['metric'] == 'centroid':
            vert_minus, horiz_minus = compute_centroid_metrics(tumor_mask_minus)
            value_minus = vert_minus
            if tumor_size_original > 0 and tumor_size_minus > 0:
                vertical_shifts_minus.append(vert_minus - vert_orig)
                horizontal_shifts_minus.append(horiz_minus - horiz_orig)

        changes_minus.append(value_minus - value_original)
        tumor_disappeared_minus.append(1 if tumor_size_minus == 0 and tumor_size_original > 0 else 0)

        # Alpha positive
        z_plus = transform_to_gmm(latent_vector, H_target, alpha=alpha_value, critical_dims=config['critical_dims'])
        pred_plus = decode_latent(z_plus)
        tumor_size_plus = (pred_plus > 0).sum()
        tumor_mask_plus = pred_plus > 0

        if config['metric'] == 'area':
            value_plus = tumor_size_plus
        elif config['metric'] == 'ratio':
            class_count_plus = (pred_plus == class_id).sum()
            value_plus = (class_count_plus / tumor_size_plus) if tumor_size_plus > 0 else 0.0
        elif config['metric'] == 'compactness':
            value_plus = compute_compactness(tumor_mask_plus) if tumor_size_plus > 0 else 0.0
        elif config['metric'] == 'centroid':
            vert_plus, horiz_plus = compute_centroid_metrics(tumor_mask_plus)
            value_plus = vert_plus
            if tumor_size_original > 0 and tumor_size_plus > 0:
                vertical_shifts_plus.append(vert_plus - vert_orig)
                horizontal_shifts_plus.append(horiz_plus - horiz_orig)

        changes_plus.append(value_plus - value_original)
        tumor_disappeared_plus.append(1 if tumor_size_plus == 0 and tumor_size_original > 0 else 0)

    # Convert to arrays
    changes_minus = np.array(changes_minus)
    changes_plus = np.array(changes_plus)
    tumor_disappeared_minus = np.array(tumor_disappeared_minus)
    tumor_disappeared_plus = np.array(tumor_disappeared_plus)

    # Calculate statistics
    results = {
        'semantic': semantic_name,
        'source_class': config['groups'][source_class],
        'target_class': config['groups'][target_class],
        'n_samples': len(source_indices),
        'critical_dims': config['critical_dims'],
        'n_critical_dims': len(config['critical_dims']),
        'metric': config['metric'],
        'alpha': alpha_value,
        'negative_alpha': {
            'mean_change': float(changes_minus.mean()),
            'std_change': float(changes_minus.std()),
            'percent_smaller': float((changes_minus < 0).mean() * 100),
            'percent_bigger': float((changes_minus > 0).mean() * 100),
            'percent_no_change': float((changes_minus == 0).mean() * 100),
            'percent_tumor_disappeared': float(tumor_disappeared_minus.mean() * 100),
            'percent_minimal_change': float((np.abs(changes_minus) < 0.01).mean() * 100)
        },
        'positive_alpha': {
            'mean_change': float(changes_plus.mean()),
            'std_change': float(changes_plus.std()),
            'percent_smaller': float((changes_plus < 0).mean() * 100),
            'percent_bigger': float((changes_plus > 0).mean() * 100),
            'percent_no_change': float((changes_plus == 0).mean() * 100),
            'percent_tumor_disappeared': float(tumor_disappeared_plus.mean() * 100),
            'percent_minimal_change': float((np.abs(changes_plus) < 0.01).mean() * 100)
        }
    }

    # Add centroid-specific metrics for laterality
    if config['metric'] == 'centroid':
        if len(vertical_shifts_minus) > 0:
            results['negative_alpha']['mean_vertical_shift'] = float(np.mean(vertical_shifts_minus))
            results['negative_alpha']['mean_horizontal_shift'] = float(np.mean(horizontal_shifts_minus))
        if len(vertical_shifts_plus) > 0:
            results['positive_alpha']['mean_vertical_shift'] = float(np.mean(vertical_shifts_plus))
            results['positive_alpha']['mean_horizontal_shift'] = float(np.mean(horizontal_shifts_plus))

    # Print results
    if config['metric'] == 'area':
        metric_name = "area"
    elif config['metric'] == 'ratio':
        metric_name = "ratio"
    elif config['metric'] == 'compactness':
        metric_name = "compactness"
    elif config['metric'] == 'centroid':
        metric_name = "centroid position (pixels)"

    print(f"\n  Alpha = -{alpha_value}:")
    print(f"    Mean change in {metric_name}: {changes_minus.mean():.4f}")
    print(f"    Std change: {changes_minus.std():.4f}")
    print(f"    % smaller: {(changes_minus < 0).mean()*100:.1f}%")
    print(f"    % bigger: {(changes_minus > 0).mean()*100:.1f}%")
    print(f"    % no change: {(changes_minus == 0).mean()*100:.1f}%")
    print(f"    % tumor disappeared: {tumor_disappeared_minus.mean()*100:.1f}%")
    if config['metric'] == 'centroid' and len(vertical_shifts_minus) > 0:
        print(f"    Mean vertical shift: {np.mean(vertical_shifts_minus):.2f} pixels (+ means down)")
        print(f"    Mean horizontal shift: {np.mean(horizontal_shifts_minus):.2f} pixels (+ means right)")

    print(f"\n  Alpha = +{alpha_value}:")
    print(f"    Mean change in {metric_name}: {changes_plus.mean():.4f}")
    print(f"    Std change: {changes_plus.std():.4f}")
    print(f"    % smaller: {(changes_plus < 0).mean()*100:.1f}%")
    print(f"    % bigger: {(changes_plus > 0).mean()*100:.1f}%")
    print(f"    % no change: {(changes_plus == 0).mean()*100:.1f}%")
    print(f"    % tumor disappeared: {tumor_disappeared_plus.mean()*100:.1f}%")
    if config['metric'] == 'centroid' and len(vertical_shifts_plus) > 0:
        print(f"    Mean vertical shift: {np.mean(vertical_shifts_plus):.2f} pixels (+ means down)")
        print(f"    Mean horizontal shift: {np.mean(horizontal_shifts_plus):.2f} pixels (+ means right)")

    return results

# -----------------------------
# Run transformations for all semantics
# -----------------------------
alpha_test = 2.0
all_results = {}

print("\n" + "="*70)
print(f"COMPREHENSIVE SEMANTIC TRANSFORMATION TEST (alpha = ±{alpha_test})")
print("="*70)

for semantic_name, config in semantic_configs.items():
    # Skip if no critical dimensions defined
    if len(config['critical_dims']) == 0:
        print(f"\nSkipping {semantic_name}: No critical dimensions defined")
        continue

    result = process_semantic(semantic_name, config, alpha_value=alpha_test)
    if result is not None:
        all_results[semantic_name] = result

# -----------------------------
# Save results
# -----------------------------
print("\n" + "="*70)
print("Saving comprehensive results...")
output_file = f"comprehensive_transformation_results_alpha{alpha_test}.pkl"
with open(output_file, "wb") as f:
    pickle.dump(all_results, f)
print(f"Results saved to {output_file}")

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for semantic, results in all_results.items():
    print(f"\n{semantic}:")
    print(f"  {results['source_class']} → {results['target_class']}")
    print(f"  Critical dims: {results['n_critical_dims']}")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Alpha -{alpha_test}: mean Δ = {results['negative_alpha']['mean_change']:.4f}")
    print(f"  Alpha +{alpha_test}: mean Δ = {results['positive_alpha']['mean_change']:.4f}")

print("\n" + "="*70)
print("All transformations complete!")