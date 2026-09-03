import logging
import os
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy.special import rel_entr
import ot
import torch.nn.functional as F
from torch import nn

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.special import rel_entr

from torch import nn

def kl_divergence_kde(real, synthetic, bandwidth=0.2, n_samples=1000):
    # approximation:
    # We have continuous multidimensional (128D) data
    # The dimensions cannot be assumed to be independent variables
    # We need some estimation -- one way is KDE

    # Fit KDEs
    kde_real = KernelDensity(bandwidth=bandwidth).fit(real)
    kde_synth = KernelDensity(bandwidth=bandwidth).fit(synthetic)
    
    # Sample from real KDE as reference points
    X_samples = real[np.random.choice(len(real), n_samples, replace=True)]
    
    # Use the KDEs that were fit above to evaluate the sampled from real
    log_p = kde_real.score_samples(X_samples)  # log P(x)
    log_q = kde_synth.score_samples(X_samples)  # log Q(x)
    
    # Technical workarounds: convert to probabilities (but I am not sure this can be done!)
    p = np.exp(log_p)
    q = np.exp(log_q)
    
    # Normalize (KL requires proper distributions)
    p /= p.sum()
    q /= q.sum()
    
    # KL(P || Q)
    kl_pq = np.sum(rel_entr(p, q))  
    
    return kl_pq

def evaluate_KL_divergence_KDE(artificial_embeddings: np.ndarray, natural_embeddings: np.ndarray):

    return kl_divergence_kde(natural_embeddings, artificial_embeddings)


def evaluate_cosine_distance(artificial_embeddings: np.ndarray, natural_embeddings: np.ndarray, output_path: str):
    """
    1) Computes the cosine distance between each pair of elements from two subsets of embeddings (artificial vs natural).
    cdist Computes distance between each pair of the two collections of inputs.
    
    2) For each artificial embedding, finds the closest (most similar) natural embedding.
    Stores for each artificial embedding, (closest_idx of natural, distance)
    
    Args:
        artificial_embeddings (np.ndarray): Array of shape (num_artificial, dim).
        natural_embeddings (np.ndarray): Array of shape (num_natural, dim).
        output_path (str): Path to the output file where results will be saved.
        
    Returns:
        float: The average minimal cosine distance across all artificial embeddings.
    """
    # Compute cosine distances (scipy returns 1 - cosine_similarity)
    cos_distances = cdist(artificial_embeddings, natural_embeddings, metric="cosine")
    
    # For each artificial embedding, find the minimal distance and its corresponding natural index
    min_indices = np.argmin(cos_distances, axis=1)
    min_distances = cos_distances[np.arange(cos_distances.shape[0]), min_indices]
    
    # Save results as lines of (closest_nat_idx, distance)
    with open(output_path, "w") as f:
        f.write("Cosine distances:\n")
        f.write(str(cos_distances.tolist()))
        f.write("\nClosest embeddings (idx, dist):\n")
        for idx, dist in zip(min_indices, min_distances):
            f.write(f"{idx}, {dist:.6f}\n")
    
    # Compute and return average minimal cosine distance
    avg_min_distance = float(np.mean(min_distances))

    return {
        "Avg. min cos distance": avg_min_distance,
        "Mean": cos_distances.mean(),
        "Variance": cos_distances.var(),
        "Min": cos_distances.min(),
        "Max": cos_distances.max(),
        "Avg. min": cos_distances.min(axis=1).mean(),
        "Avg. max": cos_distances.max(axis=1).mean()
    }


def _summarize_embeddings_(embeddings: np.ndarray):
    """
    Compute per-embedding mean and variance, then average across all embeddings.
    """
    # mean and variance per embedding (across dimensions)
    per_emb_means = embeddings.mean(axis=1)
    per_emb_vars = embeddings.var(axis=1)
    
    return {
        'mean_of_means': float(per_emb_means.mean()),
        'mean_of_variances': float(per_emb_vars.mean())
    }

def evaluate_wasserstein_distance(artificial_embeddings: np.ndarray, natural_embeddings: np.ndarray):
    # Distance matrix
    M = ot.dist(artificial_embeddings, natural_embeddings, metric='sqeuclidean')
    # Sample Weights 
    a, b = np.ones(len(artificial_embeddings)) / len(artificial_embeddings), np.ones(len(natural_embeddings)) / (len(natural_embeddings))
    # https://pythonot.github.io/all.html#ot.emd2 -- this returns number, ot.emd returns matrix
    metric = ot.emd2(a, b, M)
    return metric


def evaluate_sinkhorn_distance(artificial_embeddings: np.ndarray, natural_embeddings: np.ndarray, reg=1):
    """
    Computes the entropy-regularized Wasserstein (Sinkhorn) distance
    between two sets of embeddings.
    
    Parameters:
        artificial_embeddings: np.ndarray, shape (n_samples1, dim)
        natural_embeddings: np.ndarray, shape (n_samples2, dim)
        reg: float, regularization strength (small positive number)
    
    Returns:
        float: Sinkhorn distance between the two distributions
    """
    # Cost matrix
    M = ot.dist(artificial_embeddings, natural_embeddings, metric='sqeuclidean')
    
    # Uniform weights
    a = np.ones(len(artificial_embeddings)) / len(artificial_embeddings)
    b = np.ones(len(natural_embeddings)) / len(natural_embeddings)
    
    # Sinkhorn distance
    metric = ot.sinkhorn2(a, b, M, reg)
    
    return metric

