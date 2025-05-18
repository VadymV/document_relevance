"""
Document relevance model.

Provides functions to predict which of two documents is relevant based on
EEG attention and text semantic salience.

@author Vadym Gryshchuk (vadym.gryshchuk@proton.me)
"""

import random
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from doc_rel import MODALITY
from doc_rel.configuration.configuration import Configuration
from doc_rel.misc.utils import get_data

EPS = 1e-8


def predict_document_relevance(
    eeg_model: Union[nn.Module, BaseEstimator, Any],
    text_model: Union[BaseEstimator, Any],
    loader: DataLoader,
    conf: Configuration,
    control: bool = False,
) -> Tuple[float, int, pd.DataFrame]:
    """
    Predict which of two documents is relevant given word‐level predictions.

    Args:
        eeg_model: Trained EEG model (nn.Module) or sklearn estimator.
        text_model: Trained text salience sklearn estimator.
        loader: DataLoader yielding (eeg, sem_labels, attn_labels, words_df, embeddings).
        conf: Experiment configuration, includes conf.modality and conf.language_model.
        control: If True, randomize true labels for control experiments.

    Returns:
        acc: Accuracy comparing [doc1_true, doc2_true] vs predicted scores.
        delta: word‐count difference (doc2 – doc1).
        words_data: DataFrame with per‐word data and prediction columns added.
    """

    eeg, _, _, words_data, text_embeddings = get_data(loader)
    words_data = words_data.reset_index(drop=True)

    # Human attention model
    if conf.modality == MODALITY.TEXT:
        words_data["predicted_human_attention_soft"] = 1.0
    else:
        if isinstance(eeg_model, torch.nn.Module):
            with torch.no_grad():
                predictions_soft = (
                    torch.sigmoid(eeg_model(eeg)).cpu().numpy().ravel()
                )
        else:
            predictions_soft = eeg_model.predict_proba(eeg)[:, 1]
        words_data["predicted_human_attention_soft"] = predictions_soft

    # Semantic salience model
    if conf.modality == MODALITY.EEG:
        words_data["predicted_semantic_salience_soft"] = 1.0
    else:
        predictions_soft = text_model.predict_proba(
            text_embeddings.cpu().numpy()
        )[:, 1]
        words_data["predicted_semantic_salience_soft"] = predictions_soft

    # Similarity:
    if control:
        doc1_true_label = random.choice([0, 1])
        doc2_true_label = 1 - doc1_true_label
    else:
        doc1_true_label, doc2_true_label = 0, 1

    predicted_topical_relevance, delta = (
        _compute_document_relevance_and_delta(words_data, text_embeddings)
    )

    acc = roc_auc_score(
        [doc1_true_label, doc2_true_label], predicted_topical_relevance
    )

    return acc, delta, words_data


def _compute_document_relevance_and_delta(
    words_df: pd.DataFrame,
    word_embs: torch.Tensor,
    epsilon: float = EPS,
) -> Tuple[List[float], int]:
    """
    Core document relevance computation using predictions for attended & salient words.

    Args:
        words_df: DataFrame containing the columns
            'predicted_human_attention_soft' and 'predicted_semantic_salience_soft',
            plus boolean mask columns mask1/mask2 for doc1/doc2 membership.
        word_embs: Array of shape (N_words, D) with per-word embeddings.
        epsilon: Small constant to avoid zero weights.

    Returns:
        predictions: [1,0] or [0,1] indicating which document is predicted relevant.
        delta: number_of_words_doc2 – number_of_words_doc1
    """

    # 1) Masks for the two documents
    df = words_df.sort_values("event").reset_index(drop=True)
    mask1 = df["relevant_document"] == 0
    mask2 = df["relevant_document"] == 1

    # 2) Topic‐sanity checks
    topics = df["topic"]
    if (
        topics[mask1].nunique() != 1
        or topics[mask2].nunique() != 1
        or topics.nunique() != 2
    ):
        raise ValueError("Expected exactly two topics, one per document.")

    # 3) Get predictions as Torch tensors
    p_att = torch.from_numpy(
        df["predicted_human_attention_soft"].values
    ).float()
    p_sal = torch.from_numpy(
        df["predicted_semantic_salience_soft"].values
    ).float()

    # 4) Joint weights over *all* words
    joint = p_att * p_sal

    # 5) Per-document joint weights
    w1 = joint[mask1]
    w2 = joint[mask2]

    # 6) Compute document embeddings
    #    word_embs: (N, D)
    #    joint    : (N,)
    #    doc_all  : (D,)
    doc_all = word_embs.T @ joint

    doc1 = word_embs[mask1].T @ w1  # (D,)
    doc2 = word_embs[mask2].T @ w2  # (D,)

    # 7) Cosine similarities
    # F.cosine_similarity expects (B, D)
    sim1 = F.cosine_similarity(
        doc1.unsqueeze(0), doc_all.unsqueeze(0)
    ).item()
    sim2 = F.cosine_similarity(
        doc2.unsqueeze(0), doc_all.unsqueeze(0)
    ).item()

    # 8) Prediction and delta
    prediction = [1, 0] if sim1 >= sim2 else [0, 1]
    delta = int(mask2.sum() - mask1.sum())

    return prediction, delta
