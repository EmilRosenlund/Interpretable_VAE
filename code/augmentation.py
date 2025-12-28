import os
import shutil
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate, zoom
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import random

debug = False  # set to False for full processing
random.seed(42)
np.random.seed(42)

# Workspace

training_folder = r"C:\Users\emilr\Documents\GitHub\AAU_P7\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
validation_folder = r"C:\Users\emilr\Documents\GitHub\AAU_P7\data\BraTS2020_ValidationData"
os.makedirs(training_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# Collect NIfTI files grouped by case ID
nii_files = {}
for root, dirs, files in os.walk(training_folder):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            if not file.startswith("BraTS20_Training"):
                continue

            case_id = "_".join(file.split("_")[:3])
            if case_id not in nii_files:
                nii_files[case_id] = {}

            if "flair" in file:
                nii_files[case_id]["flair"] = os.path.join(root, file)
            elif "t1ce" in file:
                nii_files[case_id]["t1ce"] = os.path.join(root, file)
            elif "t1.nii" in file and "t1ce" not in file:
                nii_files[case_id]["t1"] = os.path.join(root, file)
            elif "t2" in file:
                nii_files[case_id]["t2"] = os.path.join(root, file)
            elif "seg" in file:
                nii_files[case_id]["seg"] = os.path.join(root, file)

case_ids = sorted(nii_files.keys())
print(f"Total cases found: {len(case_ids)}")

# === Step 1: Move 50 random cases to validation folder (skip in debug) ===
if debug:
    num_val = min(50, len(case_ids))
    val_case_ids = random.sample(case_ids, num_val)  # pick 50 random cases

    for case_id in val_case_ids:
        case_dir = os.path.dirname(list(nii_files[case_id].values())[0])
        dest_dir = os.path.join(validation_folder, os.path.basename(case_dir))
        if not os.path.exists(dest_dir):
            shutil.move(case_dir, dest_dir)

    print(f"Moved {num_val} random cases to validation folder.")
    train_case_ids = [cid for cid in case_ids if cid not in val_case_ids]
else:
    train_case_ids = case_ids[5:6]  # only one case for debug

# === Helper: Pad or crop to original shape ===
def pad_or_crop(volume, target_shape):
    padded = np.zeros(target_shape, dtype=volume.dtype)
    min_shape = np.minimum(volume.shape, target_shape)
    slices = tuple(slice(0, s) for s in min_shape)
    padded[slices] = volume[slices]
    return padded

# === Step 2: Define augmentation functions ===
def rotate_case(data, is_seg, angle=None):
    #angle = np.random.uniform(-8, 8)
    rotated_data = np.stack([
        np.ascontiguousarray(
            rotate(data[:, :, i], angle, reshape=False, order=1 if not is_seg else 0)
        ) for i in range(data.shape[2])
    ], axis=2)
    rotated_data = pad_or_crop(rotated_data, data.shape)  # ensure shape consistency
    return rotated_data, f"rotated {angle:.1f}°"

def squeeze_case(data, is_seg, factors=None):
    #factors = [np.random.uniform(0.9, 1.1), 1.0, 1.0]  # small squeeze/expand x-axis
    squeezed = zoom(data, zoom=factors, order=1 if not is_seg else 0)
    squeezed = pad_or_crop(squeezed, data.shape)
    return squeezed, "squeezed"

def brightness_case(data, is_seg, shift=None):
    if is_seg:
        return data, "no brightness change"
    #shift = np.random.randint(-30, 31)
    bright_data = np.clip(data + shift, 0, np.max(data)).astype(np.float32)
    return bright_data, f"brightness shift {shift}"

def salt_and_pepper_case(data, is_seg, rnd=None):
    if is_seg:
        return data, "no salt & pepper"
    prob = 0.01
    noisy = np.copy(data)
    #rnd = np.random.rand(*data.shape)
    noisy[rnd < prob / 2] = 0
    noisy[rnd > 1 - prob / 2] = np.max(data)
    return noisy, "salt & pepper noise"


# === Step 4: Run ===
labeling_number = len(case_ids)+1
number = 0
aug_dict = {
    1: rotate_case,
    2: squeeze_case,
    3: brightness_case,
    4: salt_and_pepper_case
}
#for i in range(len(case_ids)-number):
for i in range(2):

    #find the starting case id
    current_case_id = case_ids[number]
    # split into modalities
    modalities = list(nii_files[current_case_id].keys())
    #load data
    loaded_data = {}
    for mod in modalities:
        nii_path = nii_files[current_case_id][mod]
        nii_img = nib.load(nii_path)
        loaded_data[mod] = nii_img.get_fdata()

    # Apply all augmentations separately
    aug_params = {
        1: np.random.uniform(-8, 8),
        2: [np.random.uniform(0.85, 1.1), np.random.uniform(0.85, 1.1), 1.0],
        3: np.random.randint(-30, 31),
        4: np.random.rand(240, 240, 155)
    }
    for aug_num, aug_function in aug_dict.items():
        params = aug_params[aug_num]
        print(f"Applying augmentation {aug_num} with params {params} to case {current_case_id}")
        augmented_data = {}
        for mod in modalities:
            is_seg = (mod == "seg") # check if segmentation
            augmented_data[mod], desc = aug_function(loaded_data[mod], is_seg, params)
            print(f"Applied {desc} to {mod} of case {current_case_id}")

        if debug:
            #show augmented data
            slice_idx = 80
            fig, axes = plt.subplots(2, len(modalities), figsize=(15, 6))
            for j, mod in enumerate(modalities):
                axes[0, j].imshow(loaded_data[mod][:, :, slice_idx], cmap='gray')
                axes[0, j].set_title(f"Original {mod}")
                axes[0, j].axis('off')

                axes[1, j].imshow(augmented_data[mod][:, :, slice_idx], cmap='gray')
                axes[1, j].set_title(f"Augmented {mod} ({desc})")
                axes[1, j].axis('off')
            plt.show()

    # Save augmented cases
        new_case_id = f"BraTS20_Training_{labeling_number}_{current_case_id.replace('BraTS20_Training_', '')}"
        # create folder
        new_case_folder = os.path.join(training_folder, new_case_id)
        os.makedirs(new_case_folder, exist_ok=True)
        for mod in modalities:
            new_nii = nib.Nifti1Image(augmented_data[mod], affine=nii_img.affine)
            save_path = os.path.join(new_case_folder, f"{new_case_id}_{mod}.nii.gz")
            nib.save(new_nii, save_path)
            print(f"Saved augmented {mod} to {save_path}")

    number += 1
    labeling_number += 1