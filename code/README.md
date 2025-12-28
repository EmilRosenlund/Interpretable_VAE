# VAE-based Brain Tumor Analysis Pipeline

This repository contains a modular pipeline for analyzing brain tumor MRI data using a Variational Autoencoder (VAE) and downstream statistical and classification analyses. The codebase is designed for the BRATS dataset and supports end-to-end workflows from model training to latent space interpretation.

## Features
- **VAE Model Training:** Train a residual VAE segmenter on MRI data.
- **Latent Vector Extraction:** Encode MRI slices into latent vectors for further analysis.
- **Label Classification:** Classify latent vectors based on semantic labels.
- **Gaussian Mixture Modeling:** Fit GMMs to latent vectors for semantic clustering.
- **Critical Dimension Identification:** Identify latent dimensions most relevant to clinical labels.
- **Latent Space Transformation:** Manipulate latent vectors to study effects on generated images.

## File Overview
- `VAE_model.py` — Defines the `ResidualVAE_Segmenter` model architecture.
- `train.py` — Trains the VAE model on the BRATS dataset.
- `data_utils.py` — Data loading, preprocessing, and augmentation utilities.
- `get_latentvectors.py` — Encodes MRI slices into latent vectors using a trained VAE.
- `classify_predictions.py` — Classifies latent vectors using clinical labels (size, position).
- `fitting_gaussians.py` — Fits Gaussian Mixture Models (GMMs) to latent vectors.
- `identify_critical_dims.py` — Identifies latent dimensions critical for label separation.
- `transform_test.py` — Applies transformations in latent space and visualizes effects.

## Typical Workflow
1. **Train the VAE:**
   ```bash
   python train.py
   ```
2. **Extract Latent Vectors:**
   ```bash
   python get_latentvectors.py
   ```
3. **Classify Latent Vectors:**
   ```bash
   python classify_predictions.py
   ```
4. **Fit GMMs:**
   ```bash
   python fitting_gaussians.py
   ```
5. **Identify Critical Latent Dimensions:**
   ```bash
   python identify_critical_dims.py
   ```
6. **Transform and Visualize:**
   ```bash
   python transform_test.py
   ```

## Data & Model Requirements
- **Dataset:** The code expects the BRATS dataset in NIfTI format.
- **Labels:** Tumor size and position labels should be available or generated as described in the scripts.
- **Model Checkpoints:** Trained VAE model weights are required for encoding and transformation steps.

## Environment
- Python 3.8+
- PyTorch
- Numpy, Scikit-learn, Matplotlib, and other standard scientific libraries

Install dependencies with:
```bash
pip install torch numpy scikit-learn matplotlib nibabel
```

## Notes
- All scripts are designed to be modular and can be run independently.
- File paths for data, models, and outputs may need to be adjusted to your environment.
- GPU acceleration is supported and automatically detected.

## Citation
If you use this codebase in your research, please cite the original BRATS dataset and any relevant VAE or GMM literature.

---

For questions or contributions, please open an issue or pull request.
