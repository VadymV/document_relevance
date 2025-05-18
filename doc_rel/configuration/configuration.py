"""
This module holds the configuration of the experiment.

@author: Vadym Gryshchuk (vadym.gryshchuk@proton.me)
"""

from typing import List, Optional

import yaml

from doc_rel import EEG_MODEL, MODALITY
from doc_rel.model.embedding import LANG_MODEL


class Configuration:
    """
    Loads and holds all experiment configuration parameters.

    Attributes:
        work_dir: Working directory for outputs and logs.
        batch_size: Number of samples per batch.
        shuffle_attention_labels: Whether to shuffle human attention labels.
        use_eeg_random_features: Whether to replace EEG with random noise.
        use_text_random_features: Whether to replace text with random noise.
        language_model: Pretrained language model to use (e.g., BERT, GloVe).
        eeg_model: EEG feature‐extraction model to use (e.g., EEGNet).
        seeds: List of random seeds for reproducibility.
        users: List of user identifiers in the dataset.
        modality: Modality setting (e.g., EEG, TEXT, BIMODAL).
        dry_run: If True, run for one participant and one reading task.
        selected_seed: (Optional) Currently selected seed.
    """

    def __init__(self, conf_file: str = "./configuration.yaml") -> None:
        """
        Initialize Configuration by loading values from a YAML file.

        Args:
            conf_file: Path to the YAML configuration file.
        """
        with open(conf_file, encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)

        self.work_dir: str = settings["work_dir"]
        self.batch_size: int = settings["batch_size"]
        self.seeds: List[int] = settings["seeds"]
        self.users: List[str] = settings["users"]
        self.dry_run: bool = settings["dry_run"]

        # These can be set later via set_table_config or other methods
        self.selected_seed: Optional[int] = None
        self.modality: Optional[MODALITY] = None
        self.eeg_model: Optional[EEG_MODEL] = None
        self.shuffle_attention_labels: Optional[bool] = None
        self.use_eeg_random_features: Optional[bool] = None
        self.use_text_random_features: Optional[bool] = None
        self.language_model: Optional[LANG_MODEL] = None

    def set_table_config(
        self,
        modality: MODALITY,
        shuffle_attention_labels: bool,
        use_eeg_random_features: bool,
        use_text_random_features: bool,
        eeg_model: EEG_MODEL,
        language_model: LANG_MODEL,
    ) -> None:
        """
        Override configuration parameters for table-specific experiments.

        Args:
            modality: Which data modality to use.
            shuffle_attention_labels: Shuffle human attention labels.
            use_eeg_random_features: Replace EEG with random features.
            use_text_random_features: Replace text with random features.
            eeg_model: EEG model to use.
            language_model: Language model to use.
        """
        self.modality = modality
        self.shuffle_attention_labels = shuffle_attention_labels
        self.use_eeg_random_features = use_eeg_random_features
        self.use_text_random_features = use_text_random_features
        self.eeg_model = eeg_model
        self.language_model = language_model
