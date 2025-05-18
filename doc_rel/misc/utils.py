"""
Utility functions for data processing, metrics, and logging.

@author: Vadym Gryshchuk (vadym.gryshchuk@proton.me)
"""

import logging
import random
from pathlib import Path
from typing import Any, List, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from numpy.typing import NDArray
from scipy.stats import kendalltau, tstd
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from doc_rel import EEG_MODEL


def calculate_ratio_pos_neg(
    positive_count: int, negative_count: int, total_count: int
) -> float:
    """
    Compute the positive-to-negative sampling ratio.

    Args:
        positive_count: Number of positive samples.
        negative_count: Number of negative samples.
        total_count:    Total number of samples.

    Returns:
        The ratio (pos/total) / (neg/total).

    Raises:
        ValueError: If total_count is zero or negative_count is zero.
    """
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    if negative_count <= 0:
        raise ValueError("negative_count must be positive")
    return (positive_count / total_count) / (negative_count / total_count)


def calibrate_probability(p: float, ratio_pos_neg: float) -> float:
    """
    Calibrate a probability based on class imbalance.

    Maps threshold from skewed distribution back to 0.5.

    Args:
        p: Uncalibrated probability in [0, 1].
        ratio_pos_neg: Ratio of positive to negative class frequencies.

    Returns:
        Calibrated probability in [0, 1].

    Raises:
        ValueError: If p is not in [0, 1].
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")
    return p / (p + (1 - p) * ratio_pos_neg)


def create_folder(
    output_dir: Union[str, Path], with_checking: bool = False
) -> Path:
    """
    Create a directory if it does not exist.

    Args:
        output_dir:     Path or string of the directory to create.
        with_checking:  If True, raise an error when the directory already exists.

    Returns:
        Path object of the created (or existing) directory.

    Raises:
        ValueError: If with_checking is True and the directory already exists.
    """
    path = Path(output_dir)
    if path.exists():
        if with_checking:
            raise ValueError(f"Directory already exists: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def set_logging(log_dir: Union[str, Path], identifier: str) -> None:
    """
    Configure root logger to write to file and to console.

    Args:
        log_dir:    Directory where the log file will be saved.
        identifier: Unique identifier for this log (e.g., experiment name).
    """
    path = Path(log_dir) / f"logs_{identifier}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler(),
        ],
    )


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU and CUDA) for reproducibility.

    Args:
        seed: The random seed to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: A torch.nn.Module.

    Returns:
        Number of parameters where requires_grad is True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data(
    loader: DataLoader,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, torch.Tensor
]:
    """
    Extract and concatenate all batches from a DataLoader.

    Args:
        loader: A torch.utils.data.DataLoader yielding
                (eeg, sem_labels, attn_labels, words_df, embeddings).

    Returns:
        eeg:             Tensor of shape (N, ...).
        semantic_labels: Tensor of shape (N,).
        attention_labels:Tensor of shape (N,).
        words:          Concatenated DataFrame of length N.
        embeddings:     Tensor of shape (N, embedding_dim).
    """
    eeg_batches, sem_batches, attn_batches, words_list, emb_batches = (
        [],
        [],
        [],
        [],
        [],
    )
    for eeg, sem, attn, words, emb in loader:
        eeg_batches.append(eeg)
        sem_batches.append(sem)
        attn_batches.append(attn)
        words_list.append(words)
        emb_batches.append(emb)

    if not eeg_batches:
        raise ValueError("DataLoader yielded no batches")

    words = pd.concat(words_list, ignore_index=True)
    eeg = torch.cat(eeg_batches, dim=0)
    semantic_labels = torch.cat(sem_batches, dim=0)
    attention_labels = torch.cat(attn_batches, dim=0)
    embeddings = torch.cat(emb_batches, dim=0)

    return eeg, semantic_labels, attention_labels, words, embeddings


def get_average(values: List[float]) -> float:
    """
    Compute the arithmetic mean of a list of numbers.

    Args:
        values: List of numeric values.

    Returns:
        Mean value.

    Raises:
        ValueError: If the list is empty.
    """
    if not values:
        raise ValueError("Cannot compute average of an empty list")
    return sum(values) / len(values)


def get_sd(values: List[float]) -> float:
    """
    Compute the sample standard deviation using SciPy.

    Args:
        values: List of numeric values.

    Returns:
        Standard deviation.

    Raises:
        ValueError: If the list is empty.
    """
    if not values:
        raise ValueError("Cannot compute standard deviation of empty list")
    return float(tstd(values))


def calculate_kendall_tau(
    x: List[float], y: List[float]
) -> Tuple[float, float]:
    """
    Compute Kendall's tau correlation and log the result.

    Args:
        x: First list of scores.
        y: Second list of scores.

    Returns:
        (tau, p_value)
    """
    tau, p_value = kendalltau(x, y)
    logging.info("Kendall’s Tau: %.2f, p-value: %.2e", tau, p_value)
    return float(tau), float(p_value)


def calculate_cos_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two 1-D tensors.

    Args:
        a: Tensor of shape (dim,).
        b: Tensor of shape (dim,).

    Returns:
        Cosine similarity as a float.
    """
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return float(cos(a, b))


def get_similarity_scores(
    embeddings: torch.Tensor, reference: torch.Tensor
) -> NDArray[Any]:
    """
    Compute cosine similarity scores between each row of `embeddings`
    and the corresponding row of `reference`.

    Args:
        embeddings: Tensor of shape (N, dim).
        reference:  Tensor of shape (N, dim).

    Returns:
        Numpy array of length N with similarity scores.
    """
    if embeddings.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: embeddings {embeddings.shape} vs reference {reference.shape}"
        )
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(embeddings, reference)
    return cast(ndarray, scores.cpu().numpy())


def run_permutation_test(
    control: pd.Series,
    test: pd.Series,
    alternative: str = 'two-sided',  # 'two-sided', 'less', or 'greater'
    n_perm: int = 10000,
) -> float:
    ctrl = control.to_numpy()
    tst = test.to_numpy()
    pooled = np.concatenate([ctrl, tst])
    m = len(tst)
    obs = tst.mean() - ctrl.mean()

    diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.random.permutation(pooled)
        diffs[i] = perm[:m].mean() - perm[m:].mean()

    p_lower = np.mean(diffs < obs)
    p_upper = np.mean(diffs > obs)

    if alternative == 'less':  # test mean < control mean
        return p_lower
    elif alternative == 'greater':  # test mean > control mean
        return p_upper
    else:  # two-sided
        return min(1.0, 2 * min(p_lower, p_upper))
