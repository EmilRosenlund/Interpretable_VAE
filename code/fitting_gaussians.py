import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from numpy.linalg import det, inv
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import time
warnings.filterwarnings("ignore")


# =====================================================================
# Helper functions for multiprocessing
# =====================================================================

def fit_gmm_for_class(args):
    """
    Worker function to fit GMM for a single class.

    Parameters:
    -----------
    args : tuple
        (label_type, class_idx, X_class, label_name, n_components, max_samples_for_gmm)

    Returns:
    --------
    tuple : (label_type, class_idx, gmm_data_dict)
    """
    label_type, class_idx, X_class, label_name, n_components, max_samples_for_gmm = args

    if len(X_class) < 10:
        return (label_type, class_idx, None)

    # Subsample if too many samples
    X_class_for_fitting = X_class
    if len(X_class) > max_samples_for_gmm:
        indices = np.random.choice(len(X_class), max_samples_for_gmm, replace=False)
        X_class_for_fitting = X_class[indices]

    start_time = time.time()

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components,
                         covariance_type='diag',
                         random_state=42,
                         max_iter=100,
                         tol=1e-3,
                         verbose=0)

    gmm.fit(X_class_for_fitting)
    elapsed = time.time() - start_time

    gmm_data = {
        'gmm': gmm,
        'X': X_class,
        'n_samples': len(X_class),
        'mean': np.mean(X_class, axis=0),
        'std': np.std(X_class, axis=0),
        'bic': gmm.bic(X_class_for_fitting),
        'converged': gmm.converged_,
        'time': elapsed,
        'label_name': label_name
    }

    return (label_type, class_idx, gmm_data)


def bhattacharyya_distance(mu1, cov1, mu2, cov2):
    """Closed-form Bhattacharyya distance between two Gaussians."""
    cov_avg = 0.5 * (cov1 + cov2)
    term1 = 0.125 * np.dot((mu1 - mu2).T, np.dot(inv(cov_avg), (mu1 - mu2)))
    term2 = 0.5 * np.log(det(cov_avg) / np.sqrt(det(cov1) * det(cov2) + 1e-10))
    return np.real(term1 + term2)


def sliced_wasserstein_approx(X1, X2, n_projections=50):
    """Approximate Wasserstein distance using sliced Wasserstein."""
    distances = []
    for _ in range(n_projections):
        dir = np.random.randn(X1.shape[1])
        dir /= np.linalg.norm(dir)
        d = wasserstein_distance(X1 @ dir, X2 @ dir)
        distances.append(d)
    return np.mean(distances)


def compute_distance_pair(args):
    """
    Worker function to compute distances between two classes.

    Parameters:
    -----------
    args : tuple
        (label_type, c1, c2, gmm1_data, gmm2_data)

    Returns:
    --------
    tuple : (label_type, (c1, c2), distance_dict)
    """
    label_type, c1, c2, gmm1_data, gmm2_data = args

    m1 = gmm1_data["mean"]
    m2 = gmm2_data["mean"]

    gmm1 = gmm1_data["gmm"]
    gmm2 = gmm2_data["gmm"]

    # Weighted average of component means
    mu1 = np.average(gmm1.means_, axis=0, weights=gmm1.weights_)
    mu2 = np.average(gmm2.means_, axis=0, weights=gmm2.weights_)

    # Weighted average of component covariances
    cov1 = np.average([np.diag(c) for c in gmm1.covariances_], axis=0, weights=gmm1.weights_)
    cov2 = np.average([np.diag(c) for c in gmm2.covariances_], axis=0, weights=gmm2.weights_)

    bhat = bhattacharyya_distance(mu1, cov1, mu2, cov2)
    wass = sliced_wasserstein_approx(gmm1_data["X"], gmm2_data["X"])
    mean_dist = np.linalg.norm(m1 - m2)

    distances = {
        "Bhattacharyya": bhat,
        "Wasserstein": wass,
        "Mean": mean_dist,
    }

    return (label_type, (c1, c2), distances)


def gaussian_creation(X, labels_dict, label_names, n_components=3, max_samples_for_gmm=20000, n_workers=None):
    """
    Fit Gaussian Mixture Models for each class in each label type.

    Parameters:
    -----------
    X : np.ndarray
        Latent vectors of shape (n_samples, latent_dim)
    labels_dict : dict
        Dictionary mapping label types to label arrays
    label_names : dict
        Dictionary mapping label types to human-readable class names
    n_components : int
        Number of Gaussian components per GMM
    max_samples_for_gmm : int
        Maximum samples to use for GMM fitting (for efficiency)
    n_workers : int or None
        Number of worker processes. If None, uses cpu_count() - 2

    Returns:
    --------
    gmm_models : dict
        Nested dictionary containing fitted GMMs and statistics for each class
    distance_results : dict
        Dictionary containing pairwise distance metrics between classes
    """
    print("="*70)
    print("Gaussian Mixture Model Creation (Parallel)")
    print("="*70)

    if n_workers is None:
        n_workers = 15
    print(f"Using {n_workers} workers for parallel processing")

    # =====================================================================
    # Step 1: Prepare tasks for GMM fitting
    # =====================================================================
    print("\n[1/2] Fitting Gaussian Mixture Models for each class (parallel)...")

    tasks = []
    for label_type, labels in labels_dict.items():
        unique_classes = np.unique(labels)

        for class_idx in unique_classes:


            class_mask = labels == class_idx
            X_class = X[class_mask]
            label_name = label_names[label_type][class_idx]

            tasks.append((label_type, class_idx, X_class, label_name, n_components, max_samples_for_gmm))

    print(f"  Total tasks: {len(tasks)} GMM fits across all label types")

    # Fit GMMs in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(fit_gmm_for_class, tasks)

    # Organize results into nested dictionary
    gmm_models = {}
    for label_type, class_idx, gmm_data in results:
        if gmm_data is None:
            continue

        if label_type not in gmm_models:
            gmm_models[label_type] = {}

        gmm_models[label_type][class_idx] = gmm_data

        # Print result
        print(f"    ✓ {label_type} - Class {class_idx} ({gmm_data['label_name']}): "
              f"{gmm_data['n_samples']} samples, BIC={gmm_data['bic']:.2f}, "
              f"converged={gmm_data['converged']}, time={gmm_data['time']:.2f}s")

    # =====================================================================
    # Step 2: Compute pairwise distances (parallel)
    # =====================================================================
    print("\n[2/2] Computing pairwise distances between classes (parallel)...")

    distance_tasks = []
    for label_type in gmm_models.keys():
        classes = sorted(gmm_models[label_type].keys())

        for c1, c2 in combinations(classes, 2):
            gmm1_data = gmm_models[label_type][c1]
            gmm2_data = gmm_models[label_type][c2]
            distance_tasks.append((label_type, c1, c2, gmm1_data, gmm2_data))

    print(f"  Total distance computations: {len(distance_tasks)} pairs")

    # Compute distances in parallel
    with Pool(processes=n_workers) as pool:
        distance_results_list = pool.map(compute_distance_pair, distance_tasks)

    # Organize results
    distance_results = {}
    for label_type, (c1, c2), distances in distance_results_list:
        if label_type not in distance_results:
            distance_results[label_type] = {}
        distance_results[label_type][(c1, c2)] = distances

    for label_type in distance_results.keys():
        print(f"  ✓ Computed distances for {label_type}: {len(distance_results[label_type])} pairs")

    # =====================================================================
    # Print Variance Statistics per Semantic Class
    # =====================================================================
    print("\n" + "="*70)
    print("Class-wise Variance Statistics (for overlap analysis)")
    print("="*70)

    for label_type in sorted(gmm_models.keys()):
        print(f"\n{label_type}:")
        print("-" * 50)

        classes = sorted(gmm_models[label_type].keys())

        # Print variance for each class
        for class_idx in classes:
            gmm_data = gmm_models[label_type][class_idx]
            gmm = gmm_data['gmm']
            label_name = gmm_data['label_name']

            # Weighted average variance across GMM components
            avg_variance = np.average([np.diag(c) for c in gmm.covariances_], axis=0, weights=gmm.weights_)
            total_variance = np.sum(avg_variance)
            mean_variance = np.mean(avg_variance)
            max_variance = np.max(avg_variance)

            print(f"  Class {class_idx} ({label_name:25s}): "
                  f"Total Var={total_variance:8.2f}, Mean Var={mean_variance:6.4f}, Max Var={max_variance:6.4f}")

        # Print pairwise comparison: distance vs combined variance
        if label_type in distance_results:
            print(f"\n  Pairwise Distance vs Variance (overlap indicator):")
            for (c1, c2), distances in distance_results[label_type].items():
                gmm1 = gmm_models[label_type][c1]['gmm']
                gmm2 = gmm_models[label_type][c2]['gmm']

                # Get weighted variances
                var1 = np.average([np.diag(c) for c in gmm1.covariances_], axis=0, weights=gmm1.weights_)
                var2 = np.average([np.diag(c) for c in gmm2.covariances_], axis=0, weights=gmm2.weights_)

                # Combined std (sqrt of sum of variances) as "spread"
                combined_std = np.sqrt(np.mean(var1) + np.mean(var2))
                wasserstein_dist = distances['Wasserstein']

                # Overlap ratio: if distance < combined_std, likely overlap
                overlap_ratio = wasserstein_dist / combined_std if combined_std > 0 else float('inf')

                name1 = gmm_models[label_type][c1]['label_name']
                name2 = gmm_models[label_type][c2]['label_name']

                overlap_indicator = "OVERLAP" if overlap_ratio < 2.0 else "SEPARATED" if overlap_ratio > 3.0 else "BORDERLINE"

                print(f"    {name1:20s} vs {name2:20s}: "
                      f"Wass={wasserstein_dist:6.2f}, CombStd={combined_std:6.4f}, "
                      f"Ratio={overlap_ratio:5.2f} [{overlap_indicator}]")

    # Save results
    print("\nSaving GMM models and distance results...")
    with open("gmm_models.pkl", "wb") as f:
        pickle.dump(gmm_models, f)
    with open("distance_results.pkl", "wb") as f:
        pickle.dump(distance_results, f)
    print("  ✓ Saved gmm_models.pkl and distance_results.pkl")

    return gmm_models, distance_results


def LDA_analysis(X, labels_dict, label_names, gmm_models, distance_results):
    """
    Perform Linear Discriminant Analysis and visualize results.

    Parameters:
    -----------
    X : np.ndarray
        Latent vectors of shape (n_samples, latent_dim)
    labels_dict : dict
        Dictionary mapping label types to label arrays
    label_names : dict
        Dictionary mapping label types to human-readable class names
    gmm_models : dict
        Nested dictionary containing fitted GMMs for each class
    distance_results : dict
        Dictionary containing pairwise distance metrics between classes

    Returns:
    --------
    lda_results : dict
        Dictionary containing LDA transformers and transformed data
    """
    print("="*70)
    print("Linear Discriminant Analysis")
    print("="*70)

    # =====================================================================
    # LDA Analysis
    # =====================================================================
    print("\n[1/2] Performing LDA for each label type...")
    lda_results = {}
    for label_type, labels in labels_dict.items():
        print(f"\n  Processing {label_type}...")
        lda = LinearDiscriminantAnalysis()
        try:
            X_lda = lda.fit_transform(X, labels)
        except Exception as e:
            print(f"    Skipping {label_type} (LDA failed): {e}")
            continue

        lda_results[label_type] = (lda, X_lda)
        n_dims = X_lda.shape[1]

        # Visualization of first 2 LDA components (or 1 if only 1 available)
        plt.figure(figsize=(8, 6))

        if n_dims >= 2:
            # Plot 2D scatter
            for cls in np.unique(labels):
                idx = labels == cls
                plt.scatter(X_lda[idx, 0], X_lda[idx, 1], label=label_names[label_type][cls], s=10, alpha=0.6)
            plt.xlabel("LDA Component 1")
            plt.ylabel("LDA Component 2")
            plt.title(f"LDA Projection ({label_type}) - First 2 Components")
        else:
            # Plot 1D histogram/scatter
            for cls in np.unique(labels):
                idx = labels == cls
                # Create jittered y-values for visualization
                y_jitter = np.random.randn(np.sum(idx)) * 0.1
                plt.scatter(X_lda[idx, 0], y_jitter, label=label_names[label_type][cls], s=10, alpha=0.6)
            plt.xlabel("LDA Component 1")
            plt.ylabel("Random Jitter (for visualization)")
            plt.title(f"LDA Projection ({label_type}) - Single Component")

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"LDA_projection_{label_type}.png", dpi=150)
        plt.close()

        print(f"    ✓ Saved: LDA_projection_{label_type}.png")

    # =====================================================================
    # Visualization: Distance Heatmaps
    # =====================================================================
    print("\n[2/2] Creating distance heatmaps...")
    for label_type, pairs in distance_results.items():
        metrics = ["Bhattacharyya", "Wasserstein", "Mean"]
        classes = sorted(gmm_models[label_type].keys())
        class_labels = [label_names[label_type][c] for c in classes]

        for metric in metrics:
            mat = np.zeros((len(classes), len(classes)))
            for i, j in combinations(range(len(classes)), 2):
                c1, c2 = classes[i], classes[j]
                mat[i, j] = mat[j, i] = pairs[(c1, c2)][metric]

            # Create heatmap using matplotlib (smaller figure size)
            plt.figure(figsize=(7, 6))
            im = plt.imshow(mat, cmap='YlOrRd', aspect='auto')

            # Add colorbar with larger text
            cbar = plt.colorbar(im)
            cbar.set_label(metric, rotation=270, labelpad=20, fontsize=16)
            cbar.ax.tick_params(labelsize=14)

            # Set ticks and labels with larger font sizes
            plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha='right', fontsize=14)
            plt.yticks(range(len(class_labels)), class_labels, fontsize=14)

            # Add text annotations with larger font
            for i in range(len(class_labels)):
                for j in range(len(class_labels)):
                    text = plt.text(j, i, f'{mat[i, j]:.2f}',
                                   ha="center", va="center", fontsize=13,
                                   color="black" if mat[i, j] < mat.max()/2 else "white")

            plt.title(f"{label_type} - {metric} Matrix", fontsize=18)
            plt.tight_layout()
            plt.savefig(f"DistanceMatrix_{label_type}_{metric}.png", dpi=150)
            plt.close()

        print(f"    ✓ Saved distance matrices for {label_type}")

    print("\nSaving LDA results...")
    with open("lda_results.pkl", "wb") as f:
        pickle.dump(lda_results, f)
    print("  ✓ Saved lda_results.pkl")

    return lda_results


# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":
    print("="*70)
    print("Improved GMM + LDA Latent Space Analysis")
    print("="*70)

    # =====================================================================
    # 1. Load latent vectors and labels
    # =====================================================================

    print("\n[1/4] Loading encoded latent vectors and labels...")

    # Load latent vectors
    with open('latent_vectors.pkl', 'rb') as f:
        encoded_data = pickle.load(f)
    X = np.array([item['latent_vector'] for item in encoded_data])
    print(f"  Loaded {X.shape[0]} latent vectors of dimension {X.shape[1]}")

    # Load all label types (updated to include new labels from classify_data_doc.py)
    label_files = {
        'Positional': 'Positional_Labels.pkl',
        'Size': 'Size_Labels.pkl',
        'Ratio_1_2': 'ratio_1_2_Labels.pkl',
        'Ratio_1_3': 'ratio_1_3_Labels.pkl',
        'Ratio_2_3': 'ratio_2_3_Labels.pkl',
        'Binary': 'Binary_Labels.pkl',
        'Enhancement': 'Enhancement_Labels.pkl',
        'Necrosis': 'Necrosis_Labels.pkl',
        'Edema': 'Edema_Labels.pkl',
        'Shape': 'Shape_Labels.pkl',
        'Laterality': 'Laterality_Labels.pkl',
    }
    #/ceph/project/ce-7-723/Latent_Git/AAU_P7/labels_out/Positional_Labels.pkl
    labels_dict = {}
    for name, filepath in label_files.items():
        filepath = "/workspace/Latent_Git/AAU_P7/labels_out_predictions/" + filepath  # Adjust path as needed
        with open(filepath, 'rb') as f:
            loaded_labels = pickle.load(f)
            # Extract just the label values (third element of each tuple)
            if isinstance(loaded_labels[0], tuple):
                labels_dict[name] = np.array([item[2] for item in loaded_labels], dtype=int)
            else:
                labels_dict[name] = np.array(loaded_labels, dtype=int)
        print(f"  Loaded {name}: {len(labels_dict[name])} labels")

    # =====================================================================
    # 2. Define label names and optional dimensionality reduction
    # =====================================================================

    label_names = {
        'Positional': ["No Tumor", "Right-Up", "Right-Down", "Left-Up", "Left-Down"],
        'Size': ["No Tumor", "Small Tumor", "Medium Tumor", "Large Tumor"],
        'Ratio_1_2': ["No Tumor", "Low Ratio", "Medium Ratio", "High Ratio"],
        'Ratio_1_3': ["No Tumor", "Low Ratio", "Medium Ratio", "High Ratio"],
        'Ratio_2_3': ["No Tumor", "Low Ratio", "Medium Ratio", "High Ratio"],
        'Binary': ["No Tumor", "Tumor Present"],
        'Enhancement': ["No", "Low",
                        "Medium", "High"],
        'Necrosis': ["No Necrosis", "Low Necrosis", "Medium Necrosis", "High Necrosis"],
        'Edema': ["No Edema", "Low Edema", "Medium Edema", "High Edema"],
        'Shape': ["No Tumor", "Compact", "Intermediate", "Diffuse / Irregular"],
        'Laterality': ["No Tumor", "Left Hemisphere", "Right Hemisphere", "Bilateral / Crossing Midline"],
    }

    """ print("\n[2/4] Optional: Dimensionality Reduction for Stability")
    latent_dim = X.shape[1]
    if latent_dim > 64:
        print(f"  Reducing latent space from {latent_dim} → 32 dims via PCA for stability")
        X = PCA(n_components=32, random_state=42).fit_transform(X)
    latent_dim = X.shape[1]
    print(f"  Working with {latent_dim}-dimensional latent space") """

    # =====================================================================
    # 3. Fit GMMs and compute distances
    # =====================================================================
    print("\n[3/4] Fitting Gaussian Mixture Models and Computing Distances...")
    gmm_models, distance_results = gaussian_creation(X, labels_dict, label_names)

    # =====================================================================
    # 4. Perform LDA Analysis and Visualization
    # =====================================================================
    print("\n[4/4] Performing LDA Analysis and Creating Visualizations...")
    lda_results = LDA_analysis(X, labels_dict, label_names, gmm_models, distance_results)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)