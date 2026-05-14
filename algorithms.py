import sys
sys.path.append("/home/mauricio.alvarez/tesis/VCC/imagenetood")
from Detection.Baseline import MSP, max_logit, energy, react, ash, entropy, max_cosine
from Detection.ODIN import odin, odin_batched
from Detection.gradnorm import gradnorm
from Detection.Mahalanobis import mahalanobis
from Detection.ViM import vim, vlogits
from functools import partial
from Detection.Knn import knn
import numpy as np
# run [pip install libmr] to use openmax
#from Detection.openmax import openmax

logit_only = {
    "Maximum Softmax Probability": MSP,
    "Maximum Logits": max_logit,
    "Energy": energy,
    "Maximum Cosine": max_cosine,
    "entropy": entropy
}

gradient_based = {
    "ODIN_0.0014_1000": partial(odin, epsilon=0.0014, temperature=1000),
    "ODIN": odin,
    "ODIN_batched": odin_batched,
    "GradNorm": gradnorm,
}

training_based = {"Mahalanobis": mahalanobis, 
                  "ReAct": react, 
                  "ViM": vim, 
                  "KNN": knn,
                  #"openmax": openmax,
                  "ash": ash,
                  }

def calculate_aurra(correctness, confidences):
    """Calculates the Area Under the Rejection-Accuracy Curve (AURRA)."""
    # Sort by confidence descending
    sorted_idx = np.argsort(confidences)[::-1]
    sorted_correctness = correctness[sorted_idx]
    
    # Calculate accuracy at each coverage level
    cumulative_correct = np.cumsum(sorted_correctness)
    coverage = np.arange(1, len(correctness) + 1)
    accuracies = cumulative_correct / coverage
    
    # AURRA is the mean of accuracies across all coverage levels
    return np.mean(accuracies)

def calculate_rmsce(correctness, confidences, num_bins=15):
    """Calculates the Root Mean Square Calibration Error (RMSCE)."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries) - 1
    
    rmsce_sq = 0.0
    for b in range(num_bins):
        mask = (bin_indices == b)
        if np.sum(mask) > 0:
            bin_acc = np.mean(correctness[mask])
            bin_conf = np.mean(confidences[mask])
            bin_weight = np.sum(mask) / len(correctness)
            rmsce_sq += bin_weight * ((bin_acc - bin_conf) ** 2)
            
    return np.sqrt(rmsce_sq)