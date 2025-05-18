from enum import Enum
import torch


class EXPERIMENT(Enum):
    EEG = "eeg"
    EEG_CONTROL = "eeg-control"
    TEXT = "text"
    EEG_AND_TEXT = "eeg+text"
    RANDOM_EEG_AND_TEXT = "random_eeg+text"
    EEG_AND_RANDOM_TEXT = "eeg+random_text"


def parse_experiment(experiment_str: str) -> EXPERIMENT:
    """
    Parse a string into a EXPERIMENT enum member based on its value.

    Args:
        experiment_str: The string value (e.g. "eeg").

    Returns:
        The corresponding EXPERIMENT member.

    Raises:
        ValueError: If experiment_str does not match any EXPERIMENT value.
    """
    try:
        return EXPERIMENT(experiment_str)
    except ValueError:
        valid = ", ".join(m.value for m in EXPERIMENT)
        raise ValueError(
            f"Unknown experiment '{experiment_str}'. Valid values are: {valid}."
        )


class MODALITY(Enum):
    BIMODAL = "bimodal"
    EEG = "eeg"
    TEXT = "text"


def parse_modality(modality_str: str) -> MODALITY:
    """
    Parse a string into a MODALITY enum member based on its value.

    Args:
        modality_str: The string value (e.g. "eeg").

    Returns:
        The corresponding MODALITY member.

    Raises:
        ValueError: If modality_str does not match any MODALITY value.
    """
    try:
        return MODALITY(modality_str)
    except ValueError:
        valid = ", ".join(m.value for m in MODALITY)
        raise ValueError(
            f"Unknown modality '{modality_str}'. Valid values are: {valid}."
        )


class EEG_MODEL(Enum):
    EEGNET = "eegnet"
    LSTM = "lstm"
    LDA = "lda"
    SVM = "svm"


def parse_eeg_model(eeg_model_str: str) -> EEG_MODEL:
    """
    Parse a string into a EEG_MODEL enum member based on its value.

    Args:
        eeg_model_str: The string value (e.g. "eeg").

    Returns:
        The corresponding EEG_MODEL member.

    Raises:
        ValueError: If experiment_str does not match any EEG_MODEL value.
    """
    try:
        return EEG_MODEL(eeg_model_str)
    except ValueError:
        valid = ", ".join(m.value for m in EEG_MODEL)
        raise ValueError(
            f"Unknown EEG model '{eeg_model_str}'. Valid values are: {valid}."
        )


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
