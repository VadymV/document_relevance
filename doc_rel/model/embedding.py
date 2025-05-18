"""
Embedding models.

@author: Vadym Gryshchuk (vadym.gryshchuk@proton.me)
"""

from enum import Enum
from typing import Any, Dict, List

import gensim.downloader as api_gensim
import numpy as np
import torch
from gensim.models import KeyedVectors
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from doc_rel import get_device

# Global cache for pre-loaded models.
MODEL_CACHE = {}


class LANG_MODEL(Enum):
    """Enumeration of supported language models."""

    FAST_TEXT = "fasttext"
    WORD2VEC = "word2vec"
    GLOVE = "glove"
    BERT = "bert"
    SBERT = "sbert"


def parse_language_model(model_str: str) -> LANG_MODEL:
    """
    Parse a string into a LANG_MODEL enum member based on its value.

    Args:
        model_str: The string value of the model (e.g. "fasttext", "bert", "sbert").

    Returns:
        The corresponding LANG_MODEL member.

    Raises:
        ValueError: If model_str does not match any LANG_MODEL value.
    """
    try:
        return LANG_MODEL(model_str)
    except ValueError:
        valid = ", ".join(m.value for m in LANG_MODEL)
        raise ValueError(
            f"Unknown language model '{model_str}'. Valid values are: {valid}."
        )


def get_embeddings(
    language_model: LANG_MODEL, words: List[str], average: bool = True
) -> torch.Tensor:
    """
    Return either static or contextual embeddings for a list of words.

    Static models (FAST_TEXT, WORD2VEC, GLOVE) use _get_static_embeddings.
    Contextual models (BERT, SBERT) use _get_contextualised_embeddings.

    Args:
        language_model (LANG_MODEL): Which model to use.
        words (List[str]): Tokens to embed.
        average (bool): If True, average across tokens to a single vector.

    Returns:
        torch.Tensor: Either
            - shape (n_tokens, dim) if average=False
            - shape (1, dim) if average=True
    Raises:
        ValueError: if the model name is unsupported or input too long.
    """
    # load & cache
    if language_model not in MODEL_CACHE:
        MODEL_CACHE[language_model] = _load_model(language_model)
    entry = MODEL_CACHE[language_model]
    model, tokenizer = entry["model"], entry["tokenizer"]

    # static models
    static_keys = {
        LANG_MODEL.FAST_TEXT,
        LANG_MODEL.WORD2VEC,
        LANG_MODEL.GLOVE,
    }
    if language_model in static_keys:
        return _get_static_embeddings(model, words, average=average)

    # contextual models
    contextual_keys = {
        LANG_MODEL.BERT,
        LANG_MODEL.SBERT,
    }
    if language_model in contextual_keys:
        if len(words) > 512:
            raise ValueError("Input exceeds maximum of 512 tokens.")
        # compute per-word embeddings
        emb = _get_contextualised_embeddings(model, tokenizer, words)
        if average:
            emb = emb.mean(dim=0, keepdim=True)
        return emb

    raise ValueError(f"Unsupported language model: {language_model!r}")


def _load_model(language_model: LANG_MODEL) -> Dict[str, Any]:
    """
    Loads the specified language model and tokenizer.

    Args:
        language_model (LANG_MODEL): An enum value specifying which model to load.
                                     Supported models are:
                                     - FAST_TEXT
                                     - WORD2VEC
                                     - GLOVE
                                     - BERT
                                     - SBERT

    Returns:
        dict: A dictionary with keys:
            - "model": the loaded embedding or transformer model
            - "tokenizer": the associated tokenizer (or None for static models)

    Raises:
        ValueError: If an unsupported language model is specified.
    """
    model = None
    tokenizer = None

    if language_model == LANG_MODEL.FAST_TEXT:
        model = api_gensim.load("fasttext-wiki-news-subwords-300")
    elif language_model == LANG_MODEL.WORD2VEC:
        model = api_gensim.load("word2vec-google-news-300")
    elif language_model == LANG_MODEL.GLOVE:
        model = api_gensim.load("glove-wiki-gigaword-300")
    elif language_model == LANG_MODEL.BERT:
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif language_model == LANG_MODEL.SBERT:
        model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise ValueError("Unsupported language model.")

    return {"model": model, "tokenizer": tokenizer}


def _get_static_embeddings(
    model: Any,
    tokens: List[str],
    average: bool = True,
) -> torch.Tensor:
    """
    Retrieves static word embeddings for a list of tokens using a specified model.

    Args:
        model (KeyedVectors or object): A static embedding model such as GloVe,
            Word2Vec, or FastText. Must support either indexed access (Gensim)
            or `.get_vecs_by_tokens()` (e.g., TorchText).
        tokens (List[str]): A list of tokens (words) to embed.
        average (bool): If True, returns the mean embedding across all tokens.
                        If False, returns individual embeddings for each token.

    Returns:
        torch.Tensor: A tensor of shape (1, embedding_dim) if averaged,
                      else (n_tokens, embedding_dim).
    """
    if not tokens:
        raise ValueError("Token list is empty.")

    if isinstance(model, KeyedVectors):
        embeddings_np = np.array(
            [
                (
                    model.get_vector(token)
                    if token in model.key_to_index
                    else np.zeros(model.vector_size, dtype=np.float32)
                )
                for token in tokens
            ],
            dtype=np.float32,
        )
        embeddings = torch.from_numpy(embeddings_np)

    else:
        embeddings = model.get_vecs_by_tokens(tokens, lower_case_backup=True)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)

    if average:
        embeddings = embeddings.mean(dim=0, keepdim=True)

    return embeddings


def _get_contextualised_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    words: List[str],
) -> torch.Tensor:
    """
    Computes contextualized word embeddings from a transformer model by averaging
    subword token embeddings corresponding to each word.

    This function:
    - Concatenates input words into a sentence
    - Tokenizes while tracking character offsets
    - Uses those offsets to group subword embeddings per original word
    - Returns the mean embedding per word

    Args:
        model (PreTrainedModel): A Hugging Face transformer model (e.g., BERT).
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer for the model.
        words (List[str]): List of words (pre-tokenized) in original order.

    Returns:
        torch.Tensor: Tensor of shape (num_words, hidden_dim) containing contextualized
                      embeddings for each word, averaged over its subword tokens.
    """
    # Tokenize input and get offset mapping
    encoded_input = tokenizer(
        [" ".join(words)],
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    # Move input to model's device
    device = next(model.parameters()).device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Extract hidden states from the model
    with torch.no_grad():
        outputs = model(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
        )
    embeddings = outputs.last_hidden_state.squeeze(
        0
    )  # (seq_len, hidden_dim)
    offsets = encoded_input["offset_mapping"].squeeze(0)  # (seq_len, 2)

    # Remove special tokens ([CLS], [SEP])
    embeddings = embeddings[1:-1]
    offsets = offsets[1:-1]

    # Identify word boundaries using character offset differences
    new_word_flags = torch.empty(
        offsets.size(0), dtype=torch.bool, device=device
    )
    new_word_flags[0] = True
    new_word_flags[1:] = offsets[1:, 0] != offsets[:-1, 1]

    # Group subwords belonging to the same word
    group_ids = (
        torch.cumsum(new_word_flags, dim=0) - 1
    )  # Word indices starting at 0
    num_groups = int(group_ids[-1].item()) + 1
    hidden_dim = embeddings.size(1)

    # Sum embeddings for each group (word)
    group_sums = torch.zeros((num_groups, hidden_dim), device=device)
    group_sums = group_sums.index_add(0, group_ids, embeddings)

    # Count subword tokens per word
    group_counts = (
        torch.bincount(group_ids, minlength=num_groups).unsqueeze(1).float()
    )

    # Compute mean embedding per word
    word_embeddings = group_sums / group_counts

    return word_embeddings
