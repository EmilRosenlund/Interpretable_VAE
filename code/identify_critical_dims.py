import pickle
import numpy as np
""" Positional ⟂ Enhancement / Necrosis / Edema / Ratios / Shape
— Where a tumor sits (quadrant) doesn’t inherently change whether it enhances or necroses. Position is anatomical, enhancement is tissue biology.

Laterality ⟂ Enhancement / Necrosis / Size / Ratios
— Left/right hemisphere shouldn't determine contrast behavior or tumor area.

Binary (Tumor vs No Tumor) as a semantic label should be independent from Positional (no tumor has no meaningful position), though trivially related because no-tumor maps to positional “No Tumor”.

Ratio_1_2, Ratio_1_3, Ratio_2_3 between different pairs could be independent of Laterality and Positional. """

def compare_gaussians(gmm1 , gmm2):
    from scipy.stats import wasserstein_distance

    n_samples = 1000
    samples1, _ = gmm1.sample(n_samples)
    samples2, _ = gmm2.sample(n_samples)

    distances = []
    for dim in range(samples1.shape[1]):
        dist = wasserstein_distance(samples1[:, dim], samples2[:, dim])
        distances.append(dist)

    return np.mean(distances)

def compute_hellinger_distance_per_dim(gmm1, gmm2, n_dims):
    """
    Compute Hellinger distance between two GMMs for each latent dimension.

    For each dimension k, we compute the Hellinger distance between the
    marginal distributions of gmm1 and gmm2 on that dimension.

    The Hellinger distance between two Gaussians N(mu1, sigma1) and N(mu2, sigma2) is:
    H = sqrt(1 - sqrt((2*sigma1*sigma2)/(sigma1^2 + sigma2^2)) * exp(-0.25 * (mu1-mu2)^2 / (sigma1^2 + sigma2^2)))
    """
    distances = []

    # For each GMM, compute weighted mean and variance per dimension
    # GMM has multiple components, we compute the marginal distribution per dimension
    for dim in range(n_dims):
        # For GMM1: compute marginal mean and variance for this dimension
        # Weighted average of means
        mu1 = np.average(gmm1.means_[:, dim], weights=gmm1.weights_)
        mu2 = np.average(gmm2.means_[:, dim], weights=gmm2.weights_)

        # For diagonal covariance, extract variance for this dimension
        # Weighted average of variances (accounting for mixture)
        var1_components = gmm1.covariances_[:, dim]  # variance for each component
        var2_components = gmm2.covariances_[:, dim]

        # Mixture variance: E[Var] + Var[E]
        # E[Var] = weighted average of component variances
        # Var[E] = variance of weighted component means
        mean_var1 = np.average(var1_components, weights=gmm1.weights_)
        var_of_means1 = np.average((gmm1.means_[:, dim] - mu1)**2, weights=gmm1.weights_)
        sigma1_sq = mean_var1 + var_of_means1

        mean_var2 = np.average(var2_components, weights=gmm2.weights_)
        var_of_means2 = np.average((gmm2.means_[:, dim] - mu2)**2, weights=gmm2.weights_)
        sigma2_sq = mean_var2 + var_of_means2

        sigma1 = np.sqrt(sigma1_sq)
        sigma2 = np.sqrt(sigma2_sq)

        # Hellinger distance for Gaussians
        numerator = 2 * sigma1 * sigma2
        denominator = sigma1_sq + sigma2_sq

        if denominator < 1e-10:  # Avoid division by zero
            hellinger_dist = 0.0
        else:
            exp_term = np.exp(-0.25 * (mu1 - mu2)**2 / denominator)
            hellinger_dist = np.sqrt(1 - np.sqrt(numerator / denominator) * exp_term)

        distances.append(hellinger_dist)

    return np.array(distances)

if __name__ == "__main__":
    with open("gmm_models.pkl", "rb") as f:
        gmm_models = pickle.load(f)
    label_types = list(gmm_models.keys())
    n_labels = len(label_types)
    print(f"we have {n_labels} label types to compare.")

    # Print structure to understand what's available
    print("\nAvailable label types and their classes:")
    for label_type in label_types:
        classes = list(gmm_models[label_type].keys())
        print(f"  {label_type}: classes {classes}")

    """ gmm_models[label_type][class_idx] = {
    'gmm': <sklearn GMM object>,
    'X': <latent vectors>,
    'n_samples': <count>,
    'mean': <mean vector>,
    'std': <std vector>
    } """

    # Compare specific classes: Size (Small = class 1) with Necrosis (High = class 3)
    label_a = 'Size'
    label_b = 'Positional'
    label_c = 'Enhancement'
    label_d = 'Necrosis'
    label_e = 'Edema'
    label_f = 'Shape'
    label_g = 'Laterality'


    # Define label configurations
    label_configs = {
        'Size': {
            'irrelevant': ['Enhancement', 'Necrosis', 'Edema', 'Shape'],
            'groups': {0: "No Tumor", 1: "Small Tumor", 2: "Medium Tumor", 3: "Large Tumor"}
        },
        'Positional': {
            'irrelevant': ['Enhancement', 'Necrosis', 'Edema', 'Size', 'Shape'],
            'groups': {0: "No Tumor", 1: "Right-Up", 2: "Right-Down", 3: "Left-Up", 4: "Left-Down"}
        },
        'Enhancement': {
            'irrelevant': ['Positional', 'Size', 'Shape', 'Laterality'],
            'groups': {0: "No Contrast Enhancement", 1: "Low Contrast Enhancement", 2: "Medium Contrast Enhancement", 3: "High Contrast Enhancement"}
        },
        'Necrosis': {
            'irrelevant': ['Positional', 'Size', 'Shape', 'Laterality'],
            'groups': {0: "No Necrosis", 1: "Low Necrosis", 2: "Medium Necrosis", 3: "High Necrosis"}
        },
        'Edema': {
            'irrelevant': ['Positional', 'Size', 'Shape', 'Laterality'],
            'groups': {0: "No Edema", 1: "Low Edema", 2: "Medium Edema", 3: "High Edema"}
        },
        'Shape': {
            'irrelevant': ['Positional', 'Enhancement', 'Necrosis', 'Size', 'Laterality'],
            'groups': {0: "No Tumor", 1: "Compact", 2: "Intermediate", 3: "Diffuse / Irregular"}
        },
        'Laterality': {
            'irrelevant': ['Enhancement', 'Necrosis', 'Edema', 'Size', 'Shape'],
            'groups': {0: "No Tumor", 1: "Left Hemisphere", 2: "Right Hemisphere", 3: "Bilateral / Crossing Midline"}
        }
    }

    # Store results for all labels
    all_critical_dims = {}

    # Loop through each label type
    for current_label, config in label_configs.items():
        print("\n" + "="*70)
        print(f"Analyzing {current_label}")
        print("="*70)

        # Check if classes 1 and 2 exist for this label
        if 1 not in gmm_models[current_label] or 2 not in gmm_models[current_label]:
            print(f"Skipping {current_label}: classes 1 or 2 not available")
            continue

        # Get H_O (class 1 - Low) and H_t (class 2 - Medium)
        H_O = gmm_models[current_label][1]['gmm']
        H_t = gmm_models[current_label][2]['gmm']

        print(f"\nComparing {current_label} - {config['groups'][1]} (H_O) with {current_label} - {config['groups'][2]} (H_t):")

        # Build irrelevance set from specified semantics
        H_en = []
        semantic_names = []

        for irrelevant_label in config['irrelevant']:
            if irrelevant_label not in gmm_models:
                continue
            for cls in gmm_models[irrelevant_label]:
                if cls != 0:  # Skip "No Tumor" class
                    H_en.append(gmm_models[irrelevant_label][cls]['gmm'])
                    semantic_names.append(f"{irrelevant_label}_class_{cls}")

        print(f"Created irrelevance set with {len(H_en)} semantic classes from: {', '.join(config['irrelevant'])}")

        if len(H_en) == 0:
            print(f"Warning: No irrelevant semantics found for {current_label}, skipping")
            continue

        r""" Calculate Distributional Distance: Use the Gaussian Mixture Model (GMM) parameters
        ($\mu_k, \sigma_k$) for $H_o$, $H_t$, and every irrelevant semantic $H_j \in H_{en}$ to compute the Hellinger distance
        for that single dimension $k$.Required Distances: You need to calculate the distance between:$D_{s_k}(H_o, H_j)$
        (Original vs. Irrelevant)$D_{s_k}(H_t, H_j)$ (Target vs. Irrelevant) """

        # Get latent dimension from the GMM
        latent_dim = H_O.means_.shape[1]
        print(f"Latent space dimensionality: {latent_dim}")

        D_O_vs_H_en = np.zeros((latent_dim, len(H_en)))
        D_t_vs_H_en = np.zeros((latent_dim, len(H_en)))

        print("\nComputing Hellinger distances for each dimension and irrelevant semantic...")
        for j, H_j in enumerate(H_en):
            D_O_vs_H_en[:, j] = compute_hellinger_distance_per_dim(H_O, H_j, latent_dim)
            D_t_vs_H_en[:, j] = compute_hellinger_distance_per_dim(H_t, H_j, latent_dim)

        # Print distances for first few dimensions
        print("\n" + "="*70)
        print("Hellinger Distances between Original/Target and Irrelevant Semantics")
        print("="*70)

        for dim in range(min(6, latent_dim)):
            print(f"\nDimension {dim}:")
            for j, sem_name in enumerate(semantic_names[:5]):  # Show first 5 to avoid clutter
                print(f"  {sem_name:30s}: D_O={D_O_vs_H_en[dim, j]:.4f}, D_t={D_t_vs_H_en[dim, j]:.4f}, Δ={abs(D_t_vs_H_en[dim, j] - D_O_vs_H_en[dim, j]):.4f}")
            if len(semantic_names) > 5:
                print(f"  ... and {len(semantic_names) - 5} more irrelevant semantics")

        # Threshold for critical dimensions
        phi = 0.14

        # If both distances are below phi, consider dimension k as critical
        critical_dims = []
        for dim in range(latent_dim):
            D_O = D_O_vs_H_en[dim, :]
            D_t = D_t_vs_H_en[dim, :]
            if np.all(D_O < phi) and np.all(D_t < phi):
                critical_dims.append(dim)

        print("\n" + "="*70)
        print(f"Identified {len(critical_dims)} critical dimensions for {current_label} (phi={phi}):")
        print(critical_dims)
        print("="*70)

        # Store results
        all_critical_dims[current_label] = {
            'critical_dims': critical_dims,
            'n_critical': len(critical_dims),
            'H_O_class': config['groups'][1],
            'H_t_class': config['groups'][2],
            'irrelevant_semantics': config['irrelevant']
        }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Critical Dimensions per Semantic Label")
    print("="*70)
    for label, results in all_critical_dims.items():
        print(f"\n{label}:")
        print(f"  Comparing: {results['H_O_class']} → {results['H_t_class']}")
        print(f"  Irrelevant semantics: {', '.join(results['irrelevant_semantics'])}")
        print(f"  Critical dimensions ({results['n_critical']}): {results['critical_dims']}")

    # Save results
    with open(f"critical_dimensions_{phi}_results.pkl", "wb") as f:
        pickle.dump(all_critical_dims, f)
    print(f"\n✓ Saved results to critical_dimensions_{phi}_results.pkl")