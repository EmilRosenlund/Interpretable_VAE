import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random

# --------------------- Nifti Loader ---------------------
class NiftiLoader:
    def __init__(self, folder_path=None, validation_folder_path=None):
        self.nii_files = {}
        self.nii_files_val = {}
        if folder_path is not None:
            self.folder_path = folder_path
            #print("[DEBUG] Finding case ids from:", self.folder_path)
            #print("[DEBUG] Checking if folder exists:", os.path.exists(self.folder_path))
            if not os.path.exists(self.folder_path):
                #print("[ERROR] Folder does not exist!")
                #print("[DEBUG] Current working directory:", os.getcwd())
                #print("[DEBUG] Contents of parent directory:")
                parent_dir = os.path.dirname(self.folder_path)
                if os.path.exists(parent_dir):
                    print(os.listdir(parent_dir))
                else:
                    print("Parent directory also doesn't exist:", parent_dir)
            self.case_ids = None
            self.get_case_ids()

        else:
            print("No folder path provided, Assuming Evaluation Mode (only validation data)")
            #self.folder_path = "/ceph/project/ce-7-723/Git_2D/AAU_P7/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
        if validation_folder_path is not None:
            self.validation_folder_path = validation_folder_path
            self.case_ids_val = None
            self.get_validation_case_ids()

            #self.validation_folder_path = "/ceph/project/ce-7-723/Git_2D/AAU_P7/data/BraTS2020_ValidationData"

    def get_case_ids(self):
        #print(f"[DEBUG] Starting to walk directory: {self.folder_path}")
        file_count = 0
        nii_file_count = 0

        for root, dirs, files in os.walk(self.folder_path):
            #print(f"[DEBUG] Walking directory: {root}")
            #print(f"[DEBUG] Found {len(files)} files in this directory")

            for file in files:
                file_count += 1
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    nii_file_count += 1
                    #print(f"[DEBUG] Found .nii file: {file}")

                    if not file.startswith("BraTS20_Training") and not file.startswith("BraTS20_Validation"):
                        #print(f"[DEBUG] Skipping file (doesn't start with BraTS20_Training or BraTS20_Validation): {file}")
                        continue

                    case_id = "_".join(file.split("_")[:3])
                    #print(f"[DEBUG] Processing case_id: {case_id} from file: {file}")

                    if case_id not in self.nii_files:
                        self.nii_files[case_id] = {}

                    if "flair" in file:
                        self.nii_files[case_id]["flair"] = os.path.join(root, file)
                        #print(f"[DEBUG] Added flair for {case_id}")
                    elif "t1ce" in file:
                        self.nii_files[case_id]["t1ce"] = os.path.join(root, file)
                        #print(f"[DEBUG] Added t1ce for {case_id}")
                    elif "t1.nii" in file and "t1ce" not in file:
                        self.nii_files[case_id]["t1"] = os.path.join(root, file)
                        #print(f"[DEBUG] Added t1 for {case_id}")
                    elif "t2" in file:
                        self.nii_files[case_id]["t2"] = os.path.join(root, file)
                        #print(f"[DEBUG] Added t2 for {case_id}")
                    elif "seg" in file:
                        self.nii_files[case_id]["seg"] = os.path.join(root, file)
                        #print(f"[DEBUG] Added seg for {case_id}")

        #print(f"[DEBUG] Total files found: {file_count}")
        #print(f"[DEBUG] Total .nii files found: {nii_file_count}")
        #print(f"[DEBUG] Total cases collected: {len(self.nii_files)}")

        # Print incomplete cases for debugging
        for case_id, modalities in self.nii_files.items():
            missing = [mod for mod in ["flair","t1","t1ce","t2","seg"] if mod not in modalities]
            if missing:
                print(f"[DEBUG] Incomplete case {case_id}, missing: {missing}")

        # Keep only complete cases
        self.case_ids = sorted([k for k,v in self.nii_files.items() if all(mod in v for mod in ["flair","t1","t1ce","t2","seg"])])
        #print(f"[DEBUG] Total valid complete cases: {len(self.case_ids)}")
        if len(self.case_ids) > 0:
            print(f"[DEBUG] First few case IDs: {self.case_ids[:5]}")
        return self.nii_files

    def get_validation_case_ids(self):
        # similar to training, collect all modalities
        for root, dirs, files in os.walk(self.validation_folder_path):
            for file in files:
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    if not file.startswith("BraTS20_Validation"):
                        continue
                    case_id = "_".join(file.split("_")[:3])
                    if case_id not in self.nii_files_val:
                        self.nii_files_val[case_id] = {}
                    if "flair" in file:
                        self.nii_files_val[case_id]["flair"] = os.path.join(root, file)
                    elif "t1ce" in file:
                        self.nii_files_val[case_id]["t1ce"] = os.path.join(root, file)
                    elif "t1.nii" in file and "t1ce" not in file:
                        self.nii_files_val[case_id]["t1"] = os.path.join(root, file)
                    elif "t2" in file:
                        self.nii_files_val[case_id]["t2"] = os.path.join(root, file)
                    elif "seg" in file:
                        self.nii_files_val[case_id]["seg"] = os.path.join(root, file)

        self.case_ids_val = sorted([k for k,v in self.nii_files_val.items() if all(mod in v for mod in ["flair","t1","t1ce","t2","seg"])])
        #print(f"Total validation cases: {len(self.case_ids_val)}")
        return self.nii_files_val


# --------------------- Dataset ---------------------
class BratsDatasetBalanced(Dataset):
    def __init__(self, case_ids,
                 nii_files, target_size=(240,240),
                 tumor_slice_ratio=0.7, validate=False,
                 val_slices_per_case=3,
                 validation_tumor_ratio=0.8,
                 normalization_mode = 'mustd'):
        self.case_ids = case_ids
        self.nii_files = nii_files
        self.target_size = target_size
        self.tumor_slice_ratio = tumor_slice_ratio
        self.validation_tumor_ratio = validation_tumor_ratio  # Higher tumor ratio for validation
        self.label_map = {0:0, 1:1, 2:2, 4:3}
        self.validate = validate
        self.val_slices_per_case = val_slices_per_case
        self.norm_mode = normalization_mode

        # For validation mode: pre-compute slice indices
        if self.validate:
            if self.val_slices_per_case == -1:
                # Full comprehensive validation (all slices)
                self.all_slices = []
                print("Building comprehensive slice index for validation (ALL slices)...")
                for case_idx, case_id in enumerate(case_ids):
                    try:
                        X, y = self.load_case_multichannel(case_id)
                        num_slices = y.shape[2]
                        for slice_idx in range(num_slices):
                            self.all_slices.append((case_idx, slice_idx))
                    except Exception as e:
                        print(f"Warning: Could not load case {case_id}: {e}")
                print(f"Total slices for validation: {len(self.all_slices)} across {len(case_ids)} cases")
            else:
                # Balanced fixed validation (consistent subset)
                self.all_slices = []
                print(f"Building balanced validation index ({self.val_slices_per_case} slices per case, {self.validation_tumor_ratio:.1%} tumor ratio)...")
                rng = rng = np.random.default_rng(42)  # Fixed seed for reproducible validation
                for case_idx, case_id in enumerate(case_ids):
                    try:
                        X, y = self.load_case_multichannel(case_id)
                        tumor_slices = [i for i in range(y.shape[2]) if np.any(y[:,:,i] != 0)]
                        all_slices = list(range(y.shape[2]))

                        selected_slices = []
                        # Select mix of tumor and non-tumor slices using validation tumor ratio
                        for _ in range(self.val_slices_per_case):
                            if len(tumor_slices) > 0 and rng.uniform(0,1) < self.validation_tumor_ratio:
                                slice_idx = np.random.choice(tumor_slices)
                                tumor_slices.remove(slice_idx)  # Don't select same slice twice
                            else:
                                slice_idx = np.random.choice(all_slices)
                                all_slices.remove(slice_idx)
                            selected_slices.append(slice_idx)
                            self.all_slices.append((case_idx, slice_idx))
                    except Exception as e:
                        print(f"Warning: Could not load case {case_id}: {e}")
                print(f"Total slices for validation: {len(self.all_slices)} ({self.val_slices_per_case} per case, {len(case_ids)} cases)")

    def __len__(self):
        if self.validate:
            return len(self.all_slices)
        else:
            return len(self.case_ids)

    @staticmethod
    def get_tumor_slices(y_volume):
        """Return indices of slices containing tumor"""
        return [i for i in range(y_volume.shape[2]) if np.any(y_volume[:,:,i] != 0)]

    def __getitem__(self, idx):
        if self.validate:
            # Validation mode: use pre-computed (case_idx, slice_idx) pairs
            case_idx, slice_idx = self.all_slices[idx]
            case_id = self.case_ids[case_idx]
        else:
            # Training mode: random slice selection
            case_id = self.case_ids[idx]

        try:
            X, y = self.load_case_multichannel(case_id)
        except FileNotFoundError as e:
            #print(f"Missing file for {case_id}: {e}, skipping")
            # pick next case
            return self.__getitem__((idx+1) % len(self))

        if not self.validate:
            # ---------------- Training: Balanced slice selection ---------------- # with random slice picking
            tumor_slices = self.get_tumor_slices(y)
            all_slices = list(range(y.shape[2]))
            if len(tumor_slices) > 0 and np.random.rand() < self.tumor_slice_ratio:
                slice_idx = np.random.choice(tumor_slices)
            else:
                slice_idx = np.random.choice(all_slices)

        # ---------------- Extract slice ----------------
        X_slice = X[:,:,slice_idx,:].astype(np.float32)
        y_slice = y[:,:,slice_idx]
        y_slice = np.vectorize(self.label_map.get)(y_slice, 0)

        match self.norm_mode:
            case 'mustd':
                # Normalize each modality with zero-mean, unit variance (ignoring zeros)
                for c in range(X_slice.shape[2]):  # For each modality/channel
                    modality = X_slice[:, :, c]
                    # Create mask to ignore zero values (background)
                    mask = modality != 0
                    if np.any(mask):  # Only normalize if there are non-zero values
                        mean_val = np.mean(modality[mask])
                        std_val = np.std(modality[mask])
                        if std_val > 1e-6:  # Avoid division by zero
                            X_slice[:, :, c] = np.where(mask, (modality - mean_val) / std_val, 0)
                        else:
                            X_slice[:, :, c] = np.where(mask, modality - mean_val, 0)

            case 'tanh':
                for c in range(X_slice.shape[2]):  # For each modality/channel
                    modality = X_slice[:, :, c]
                    # Create mask to ignore zero values (background)
                    mask = modality != 0
                    if np.any(mask):  # Only normalize if there are non-zero values
                        mean_val = np.mean(modality[mask])
                        std_val = np.std(modality[mask])
                        if std_val > 1e-6:  # Avoid division by zero
                            X_slice[:, :, c] = np.where(mask, 0.5 * np.tanh(0.01*(modality - mean_val) / std_val), 0)
                        else:
                            X_slice[:, :, c] = np.where(mask, modality - mean_val, 0)

            case 'minmax':
                epsilon = 1e-8
                for c in range(X_slice.shape[2]):  # For each modality/channel
                    modality = X_slice[:, :, c]
                    min_val = np.min(modality)
                    max_val = np.max(modality)

                    X_slice[:, :, c] = (modality - min_val) / (max_val - min_val + epsilon)

        # Convert to tensors
        X_slice = torch.from_numpy(X_slice).permute(2,0,1).unsqueeze(0)
        y_slice = torch.from_numpy(y_slice).unsqueeze(0).unsqueeze(0)

        # Resize
        X_slice = F.interpolate(X_slice, size=self.target_size, mode="bilinear", align_corners=False)
        y_slice = F.interpolate(y_slice.float(), size=self.target_size, mode="nearest")

        X_slice = X_slice.squeeze(0)
        y_slice = y_slice.squeeze(0).squeeze(0).long()
        return X_slice, y_slice, case_id

    def load_case_multichannel(self, case_id):
        """Load all modalities as (H,W,D,C) and segmentation as (H,W,D)"""
        #print(f"[DEBUG] Loading case: {case_id}")
        modalities = ["flair","t1","t1ce","t2"]
        channels = []

        for mod in modalities:
            path = self.nii_files[case_id][mod]
            #print(f"[DEBUG] Loading {mod} from: {path}")
            #print(f"[DEBUG] File exists: {os.path.exists(path)}")

            if not os.path.exists(path):
                #print(f"[ERROR] File not found: {path}")
                raise FileNotFoundError(path)

            try:
                img = nib.load(path).get_fdata().astype(np.float32)
                #print(f"[DEBUG] Loaded {mod} with shape: {img.shape}")
                channels.append(img)
            except Exception as e:
                #print(f"[ERROR] Failed to load {mod}: {e}")
                raise

        multichannel_img = np.stack(channels, axis=-1)
        #print(f"[DEBUG] Multichannel image shape: {multichannel_img.shape}")

        seg_path = self.nii_files[case_id]["seg"]
        #print(f"[DEBUG] Loading segmentation from: {seg_path}")
        #print(f"[DEBUG] Segmentation file exists: {os.path.exists(seg_path)}")

        seg = nib.load(seg_path).get_fdata().astype(np.int16)
        #print(f"[DEBUG] Segmentation shape: {seg.shape}")

        return multichannel_img, seg


# --------------------- Example usage ---------------------
if __name__ == "__main__":
    folder_path = "/ceph/project/ce-7-723/Git_2D/AAU_P7/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    loader = NiftiLoader(folder_path)
    #print("Valid cases:", loader.case_ids)

    dataset = BratsDatasetBalanced(loader.case_ids, loader.nii_files)
    #print("Dataset length:", len(dataset))

    # Test load
    X, y = dataset[0]
    #print("X shape:", X.shape, "y shape:", y.shape)