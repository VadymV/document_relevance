"""
This module holds the Collator, DataLoader, and DataProvider.

@author: Vadym Gryshchuk (vadym.gryshchuk@proton.me)
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler

from doc_rel import EEG_MODEL
from doc_rel.configuration.configuration import Configuration
from doc_rel.model.embedding import LANG_MODEL, get_embeddings

NUM_WORKERS = 2
PIN_MEMORY = False
PERSISTENT_WORKERS = False


class Collator:
    """
    Turn a list of (eeg, word_df, emb) into batched tensors:
      - eeg_batch: Tensor[B,...]
      - sem_labels: Tensor[B]
      - attn_labels: Tensor[B]
      - words_df:   concat'd DataFrame
      - embs:       Tensor[B, D]
    """

    def __init__(
        self,
        eeg_model: EEG_MODEL,
        use_eeg_random: bool = False,
        use_text_random: bool = False,
    ):
        self.eeg_model = eeg_model
        self.use_eeg_random = use_eeg_random
        self.use_text_random = use_text_random

    def __call__(
        self,
        batch: List[Tuple[Tensor, pd.DataFrame, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, pd.DataFrame, Tensor]:
        if not batch:
            raise ValueError("Empty batch")

        eegs, dfs, embs = zip(*batch)
        eeg_batch = torch.stack(eegs).float()
        if self.use_eeg_random:
            eeg_batch = torch.randn_like(eeg_batch)

        # reshape for EEG_MODEL
        if self.eeg_model in (EEG_MODEL.LDA, EEG_MODEL.SVM):
            eeg_batch = eeg_batch.flatten(start_dim=1)
        elif self.eeg_model == EEG_MODEL.EEGNET:
            eeg_batch = eeg_batch.unsqueeze(1)
        elif self.eeg_model == EEG_MODEL.LSTM:
            eeg_batch = eeg_batch.permute(0, 2, 1)
        else:
            raise ValueError(f"Unsupported model {self.eeg_model}")

        words_df = pd.concat(dfs, ignore_index=True)
        sem = torch.from_numpy(
            words_df["semantic_relevance"].to_numpy()
        ).float()
        att = torch.from_numpy(
            words_df["attention_labels"].to_numpy()
        ).float()

        emb_batch = torch.vstack(embs).float()
        if self.use_text_random:
            emb_batch = torch.randn_like(emb_batch)

        return eeg_batch, sem, att, words_df, emb_batch


class BatchDataLoader:
    """
    Wraps TorchDataLoader with our Collator.
    """

    def __init__(
        self,
        batch_size: int,
        eeg_model: EEG_MODEL,
        use_eeg_random: bool = False,
        use_text_random: bool = False,
        num_workers: int = NUM_WORKERS,
        persistent_workers: bool = PERSISTENT_WORKERS,
        pin_memory: bool = PIN_MEMORY,
    ) -> None:
        self.batch_size = batch_size
        self.collate_fn = Collator(
            eeg_model, use_eeg_random, use_text_random
        )
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

    def create_loader(
        self, dataset: Dataset, sampler: Sampler
    ) -> TorchDataLoader:
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class DataProvider(Dataset):
    """
    Loads per‐document EEG + word data, builds per‐word samples.
    """

    CHUNKS_DEFAULT = 7
    CHUNKS_EEGNET = 151

    def __init__(
        self,
        dir_data: str,
        conf: Configuration,
        user: List[str],
        language_model: Optional[LANG_MODEL] = None,
        eegnet: bool = False,
    ):
        self.dir_data = dir_data
        self.conf = conf
        self.users = set(user)
        self.language_model = language_model
        self.chunks = self.CHUNKS_EEGNET if eegnet else self.CHUNKS_DEFAULT

        # load & preprocess
        self.documents = self._find_docs()
        if not self.documents:
            raise FileNotFoundError(
                f"No docs in {dir_data} for users {user}"
            )

        self.data = self._load_all()

    def _find_docs(self) -> List[str]:
        return [
            f
            for f in os.listdir(self.dir_data)
            if any(u in f for u in self.users)
        ]

    def _load_all(self) -> Dict[str, Any]:
        eeg_tensors, dfs, emb_batches = [], [], []

        for doc in self.documents:
            path = os.path.join(self.dir_data, doc)
            eeg_np, text_df = self._load_single(path, doc)

            # chunk & mean
            splits = np.array_split(eeg_np, self.chunks, axis=-1)
            means = [torch.from_numpy(s.mean(axis=-1)) for s in splits]
            eeg_t = torch.stack(means, dim=-1)  # (N, channels, chunks)

            # annotate
            m = re.search(r"(TRPB)\d{3}", doc)
            df = text_df.copy()
            df["user"] = m.group(0)
            df["relevant_document"] = df["topic"] == df["selected_topic"]
            df["attention_labels"] = (
                df["semantic_relevance"] & df["relevant_document"]
            ).astype(int)

            # embeddings in one shot
            words = df["word"].str.replace(" ", "", regex=False).tolist()
            emb = get_embeddings(self.language_model, words, average=False)

            eeg_tensors.append(eeg_t)
            dfs.append(df)
            emb_batches.append(emb)

        # concatenate per-word
        words_df = pd.concat(dfs, ignore_index=True)
        all_eeg = torch.vstack(eeg_tensors)
        all_emb = torch.vstack(emb_batches)

        return {"eeg": all_eeg, "text": words_df, "embeddings": all_emb}

    def __len__(self) -> int:
        return len(self.data["text"])

    def __getitem__(self, idx: int) -> Tuple[Tensor, pd.DataFrame, Tensor]:
        return (
            self.data["eeg"][idx],
            self.data["text"].iloc[[idx]],
            self.data["embeddings"][idx],
        )

    def _load_single(
        self, folder: str, name: str
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        eeg_path = os.path.join(folder, f"{name}.npy")
        txt_path = os.path.join(folder, f"{name}.pkl")
        if not (os.path.exists(eeg_path) and os.path.exists(txt_path)):
            raise FileNotFoundError(f"Missing {name}")
        return np.load(eeg_path), pd.read_pickle(txt_path)


class ParticipantDataProvider(DataProvider):
    """
    Like DataProvider, but also assigns 8 “Block” groups based on the `event` column.
    """

    def __init__(
        self, dir_data, conf, user, language_model=None, eegnet=False
    ):
        super().__init__(dir_data, conf, user, language_model, eegnet)
        self.topic_blocks = self._assign_blocks()

    def _assign_blocks(self) -> None:
        unique_topics = self.data['text']['topic'].unique()
        if len(unique_topics) != 16:
            raise ValueError(
                f"Expected 16 unique topics, but found {len(unique_topics)}"
            )

        blocks = (
            self.data["text"][
                ["topic", "selected_topic", "relevant_document", "event"]
            ]
            .groupby(by=["event"])
            .min()
            .drop_duplicates()
            .sort_index()
        )
        blocks["Block"] = np.repeat([*range(1, 9)], 2).tolist()
        blocks = blocks[["topic", "Block"]].drop_duplicates()
        self.data["text"]["Block"] = self.data["text"].apply(
            lambda row: blocks[blocks["topic"] == row["topic"]][
                "Block"
            ].item(),
            axis=1,
        )

        groups = self.data["text"]["Block"].tolist()
        if len(np.unique(groups)) != 8:
            raise ValueError("Eight reading trials are expected.")

        return self.data["text"]["Block"].tolist()
