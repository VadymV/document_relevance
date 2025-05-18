"""
Runs all experiments.

@author: Vadym Gryshchuk (vadym.gryshchuk@proton.me)
"""

"""
Runs all experiments.

@author: Vadym Gryshchuk
"""

import argparse
import copy
import logging
import os
import ssl
from typing import Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from torch.utils.data import SubsetRandomSampler
from doc_rel.model.embedding import LANG_MODEL, parse_language_model

from doc_rel.configuration.configuration import Configuration
from doc_rel.data_operations.loader import (
    BatchDataLoader,
    ParticipantDataProvider,
)
from doc_rel import (
    EEG_MODEL,
    EXPERIMENT,
    MODALITY,
    parse_eeg_model,
    parse_experiment,
)
from doc_rel.misc.utils import (
    create_folder,
    get_average,
    get_data,
    set_logging,
    set_seed,
)
from doc_rel.model.doc_rel import predict_document_relevance
from doc_rel.model.eegnet import EEGNet
from doc_rel.model.lstm import LSTM
from doc_rel.model.trainer import ClassifierTrainer
from doc_rel.misc.utils import calculate_ratio_pos_neg, calibrate_probability

# disable SSL cert verification if needed
ssl._create_default_https_context = ssl._create_unverified_context


def _make_eeg_model(
    model_name: EEG_MODEL,
) -> Union[torch.nn.Module, BaseEstimator]:
    """
    Factory: returns a fresh, untrained EEG model instance.

    Args:
        model_name: The EEG_MODEL enum specifying which model to build.

    Returns:
        A sklearn estimator (LDA or SVC) or a torch.nn.Module (EEGNet or LSTM).

    Raises:
        ValueError: If the model_name is not recognized.
    """
    factory = {
        EEG_MODEL.LDA: lambda: LinearDiscriminantAnalysis(
            shrinkage="auto", solver="lsqr"
        ),
        EEG_MODEL.SVM: lambda: SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True
        ),
        EEG_MODEL.EEGNET: lambda: EEGNet(
            chunk_size=151, num_electrodes=32, num_classes=1
        ),
        EEG_MODEL.LSTM: lambda: LSTM(
            input_dim=32, hid_channels=32, num_classes=1
        ),
    }
    try:
        return factory[model_name]()
    except KeyError:
        raise ValueError(f"Unknown EEG model: {model_name!r}")


def fit_word_models(
    train_loader: BatchDataLoader,
    val_loader: BatchDataLoader,
    shuffle_attention_labels: bool,
    eeg_model_name: EEG_MODEL,
    root_dir: str,
    modality: MODALITY,
) -> Tuple[
    Optional[Union[torch.nn.Module, BaseEstimator]],
    Optional[BaseEstimator],
]:
    """
    Train word-level EEG and text salience models.

    Args:
        train_loader: yields (eeg, semantic_labels, attention_labels, _, embeddings).
        val_loader: validation loader with same output.
        shuffle_attention_labels: if True, randomize attention labels.
        eeg_model_name: which EEG_MODEL to use.
        root_dir: directory for checkpoints/logs.
        modality: whether to train EEG, TEXT, or BIMODAL models.

    Returns:
        Tuple[eeg_model, text_model], either may be None if skipped.
    """
    eeg_data, sem_labels, attn_labels, _, embeddings = get_data(train_loader)
    if sem_labels.numel() == 0 or embeddings.numel() == 0:
        raise ValueError("No training data provided to fit_word_models")

    # shuffle labels if requested
    if shuffle_attention_labels:
        perm = torch.randperm(attn_labels.numel(), device=attn_labels.device)
        attn_labels = attn_labels.view(-1)[perm].view_as(attn_labels)

    eeg_model, text_model = None, None

    # EEG branch
    if modality in (MODALITY.EEG, MODALITY.BIMODAL):
        model = _make_eeg_model(eeg_model_name)
        if isinstance(model, torch.nn.Module):
            early_stop = EarlyStopping(
                monitor="val_loss", mode="min", patience=1, verbose=True
            )
            trainer = ClassifierTrainer(
                model,
                shuffle_targets=shuffle_attention_labels,
                root_dir=root_dir,
            )
            trainer.fit(train_loader, val_loader, callbacks=[early_stop])
            eeg_model = trainer.model
        else:
            model.fit(eeg_data, attn_labels.cpu().numpy())
            eeg_model = model

    # TEXT branch
    if modality in (MODALITY.TEXT, MODALITY.BIMODAL):
        lda = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
        lda.fit(embeddings.cpu().numpy(), sem_labels.cpu().numpy())
        text_model = lda

    return eeg_model, text_model


def predict_word_models(
    eeg_model: Union[torch.nn.Module, BaseEstimator, Any, None],
    text_model: Union[BaseEstimator, Any, None],
    loader: BatchDataLoader,
    ratio_pos_neg_sem: float,
    ratio_pos_neg_attn: float,
    log: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict word relevance and compute AUCs for EEG and text models.

    Args:
        eeg_model: trained EEG model or None.
        text_model: trained text model or None.
        loader: DataLoader for test data.
        log: if True, log the computed AUCs.

    Returns:
        (auroc_eeg, auroc_text,
         true_attention, true_semantic,
         pred_attention, pred_semantic)
    """
    eeg_data, sem_labels, attn_labels, _, embeddings = get_data(loader)

    # EEG predictions
    if eeg_model is not None:
        if isinstance(eeg_model, torch.nn.Module):
            with torch.no_grad():
                preds_attn = (
                    torch.sigmoid(eeg_model(eeg_data)).cpu().numpy().ravel()
                )
        else:
            preds_attn = eeg_model.predict_proba(eeg_data.cpu().numpy())[
                :, 1
            ]
        preds_attn = np.array(
            [
                calibrate_probability(p, ratio_pos_neg_attn)
                for p in preds_attn
            ]
        )
        auroc_eeg = roc_auc_score(attn_labels.cpu().numpy(), preds_attn)
    else:
        preds_attn = np.array([])
        auroc_eeg = float("nan")

    # Text predictions
    if text_model is not None:
        preds_text = text_model.predict_proba(embeddings.cpu().numpy())[:, 1]
        preds_text = np.array(
            [calibrate_probability(p, ratio_pos_neg_sem) for p in preds_text]
        )
        auroc_text = roc_auc_score(sem_labels.cpu().numpy(), preds_text)
    else:
        preds_text = np.array([])
        auroc_text = float("nan")

    if log:
        logging.info("AUC-EEG: %.3f | AUC-Text: %.3f", auroc_eeg, auroc_text)

    return (
        auroc_eeg,
        auroc_text,
        attn_labels.cpu().numpy(),
        sem_labels.cpu().numpy(),
        preds_attn,
        preds_text,
    )


def run(
    configuration: Configuration,
    participant: str,
    metrics: dict,
) -> pd.DataFrame:
    """
    Execute experiment for one participant over Leave-One-Group-Out CV.

    Args:
        configuration: loaded Configuration object.
        participant: participant ID string.
        metrics: dict to accumulate trial-level metrics.

    Returns:
        DataFrame of word-level predictions across folds.
    """
    run_conf = copy.deepcopy(configuration)
    data_dir = os.path.join(run_conf.work_dir, "data_prepared_for_benchmark")

    provider = ParticipantDataProvider(
        dir_data=data_dir,
        conf=run_conf,
        user=[participant],
        language_model=run_conf.language_model,
        eegnet=(run_conf.eeg_model == EEG_MODEL.EEGNET),
    )
    groups = provider.topic_blocks
    loader_factory = BatchDataLoader(
        batch_size=run_conf.batch_size,
        use_eeg_random=run_conf.use_eeg_random_features,
        use_text_random=run_conf.use_text_random_features,
        eeg_model=run_conf.eeg_model,
    )

    logo = LeaveOneGroupOut()
    word_dfs = []
    doc_accs, eeg_aucs = [], []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(range(len(provider)), groups=groups), start=1
    ):
        # Determine validation group
        unique_test = np.unique(np.array(groups)[test_idx])
        if len(unique_test) != 1:
            raise ValueError("Test fold must contain exactly one group.")
        test_group = unique_test[0]
        all_groups = np.unique(groups)
        val_group = all_groups[(all_groups == test_group).argmax() - 1]

        # Split indices
        val_idx = train_idx[np.array(groups)[train_idx] == val_group]
        train_idx = np.setdiff1d(train_idx, val_idx)

        for arr in (train_idx, val_idx, test_idx):
            if len(arr) == 0:
                raise ValueError("Empty train/val/test split detected.")

        # Build samplers and loaders
        train_loader = loader_factory.create_loader(
            provider, sampler=SubsetRandomSampler(train_idx)
        )
        val_loader = loader_factory.create_loader(
            provider, sampler=SubsetRandomSampler(val_idx)
        )
        test_loader = loader_factory.create_loader(
            provider, sampler=SubsetRandomSampler(test_idx)
        )

        # Fit word‐level models
        eeg_model, text_model = fit_word_models(
            train_loader=train_loader,
            val_loader=val_loader,
            shuffle_attention_labels=run_conf.shuffle_attention_labels,
            eeg_model_name=run_conf.eeg_model,
            root_dir=run_conf.work_dir,
            modality=run_conf.modality,
        )

        # Compute the positive-to-negative sampling ratio
        _, sem_labels, attn_labels, _, _ = get_data(train_loader)
        ratio_pos_neg_sem = calculate_ratio_pos_neg(
            positive_count=sem_labels.sum(),
            negative_count=len(sem_labels) - sem_labels.sum(),
            total_count=len(sem_labels),
        )
        ratio_pos_neg_attn = calculate_ratio_pos_neg(
            positive_count=attn_labels.sum(),
            negative_count=len(attn_labels) - attn_labels.sum(),
            total_count=len(attn_labels),
        )

        # Word‐level predictions
        auc_eeg, auc_text, attn_lbl, sem_lbl, pred_attn, pred_sem = (
            predict_word_models(
                eeg_model=eeg_model,
                text_model=text_model,
                loader=test_loader,
                ratio_pos_neg_sem=ratio_pos_neg_sem,
                ratio_pos_neg_attn=ratio_pos_neg_attn,
                log=True,
            )
        )
        metrics["auc_eeg"].append(auc_eeg)
        metrics["auc_text"].append(auc_text)
        metrics["attention_labels"].append(attn_lbl)
        metrics["semantic_salience_labels"].append(sem_lbl)
        metrics["predicted_attention"].append(pred_attn)
        metrics["predicted_semantic_salience"].append(pred_sem)

        # Document‐level prediction
        pred_doc, delta, word_df = predict_document_relevance(
            eeg_model=eeg_model,
            text_model=text_model,
            loader=test_loader,
            conf=run_conf,
        )
        metrics["predicted_topical_document_relevance"].append(pred_doc)
        metrics["delta"].append(delta)

        # Track fold‐level metrics
        metrics["user"].append(participant)
        metrics["seed"].append(run_conf.selected_seed)
        doc_accs.append(pred_doc)
        eeg_aucs.append(auc_eeg)

        # Annotate and collect word‐df
        word_df["user"] = participant
        word_df["fold"] = fold
        word_df["seed"] = run_conf.selected_seed
        word_dfs.append(word_df)

    logging.info(
        "Document accuracy=%.3f | EEG AUC=%.3f \n",
        get_average(doc_accs),
        get_average(eeg_aucs),
    )
    return pd.concat(word_dfs, ignore_index=True)


def run_experiment(conf: Configuration, experiment_id: str) -> None:
    """
    Loop over seeds and participants, accumulate and save results.

    Args:
        conf: base Configuration.
        experiment_id: identifier used in output filenames.
    """
    exp_conf = copy.deepcopy(conf)
    results = {
        k: []
        for k in (
            "user",
            "seed",
            "auc_eeg",
            "auc_text",
            "attention_labels",
            "semantic_salience_labels",
            "predicted_attention",
            "predicted_semantic_salience",
            "predicted_topical_document_relevance",
            "delta",
        )
    }
    logging.info("Running: %s", experiment_id)

    all_word_dfs = []

    for seed in range(1, exp_conf.seeds + 1):
        logging.info("Seed: %s", seed)
        set_seed(seed)
        exp_conf.selected_seed = seed
        for user in exp_conf.users:
            logging.info("Participant: %s", user)
            df_words = run(exp_conf, user, metrics=results)
            all_word_dfs.append(df_words)
            if exp_conf.dry_run:
                break
        if exp_conf.dry_run:
            break

    # Save word‐level predictions
    out_dir = os.path.join(conf.work_dir, "predictions")
    create_folder(out_dir)
    pd.concat(all_word_dfs, ignore_index=True).to_csv(
        os.path.join(out_dir, f"{experiment_id}.csv"), index=False
    )

    # Save trial‐level metrics
    df_metrics = pd.DataFrame(results)
    df_metrics["participant"] = df_metrics["user"].str[-2:]
    df_metrics.to_pickle(os.path.join(out_dir, f"{experiment_id}.pkl"))


def run_table_experiment(
    conf: Configuration,
    experiment: EXPERIMENT,
    eeg_model: EEG_MODEL,
    language_model: LANG_MODEL,
    use_eeg_model: bool,
    use_language_model: bool,
) -> None:
    """
    Configure and run an experiment variant based on which modalities are active.

    Args:
        conf: base Configuration.
        experiment: EXPERIMENT enum for naming.
        eeg_model: EEG_MODEL enum for naming.
        language_model: LANG_MODEL enum for naming.
        use_eeg_model: if True, include EEG modality.
        use_language_model: if True, include language modality.
    """
    if not (use_eeg_model or use_language_model):
        raise ValueError(
            "At least one of use_eeg_model or use_language_model must be True."
        )

    eid = f"{experiment.value}"
    if use_language_model:
        eid += f"#language_model-{language_model.value}"
    if use_eeg_model:
        eid += f"#eeg_model-{eeg_model.value}"

    run_experiment(conf, eid)


def main():
    """
    Parse CLI arguments and dispatch to the appropriate experiment runner.
    """
    parser = argparse.ArgumentParser("Run Document Relevance Experiments")
    parser.add_argument(
        "--eeg_model",
        type=parse_eeg_model,
        default=EEG_MODEL.LDA,
        help='EEG model: "lstm", "eegnet", "lda", or "svm".',
    )
    parser.add_argument(
        "--language_model",
        type=parse_language_model,
        default=LANG_MODEL.SBERT,
        help='Language model: "fasttext", "word2vec", "glove", "bert", or "sbert".',
    )
    parser.add_argument(
        "--experiment",
        type=parse_experiment,
        default=EXPERIMENT.EEG_AND_TEXT,
        help="Experiment to run: eeg, eeg-control, text, eeg+text, eeg+random_text, random_eeg+text.",
    )
    args = parser.parse_args()

    conf = Configuration()
    set_logging(
        conf.work_dir,
        f"{args.experiment.value}_{args.language_model.value}_{args.eeg_model.value}",
    )

    # Dispatch each experiment variant
    variants = [
        # (use_eeg_model, use_lang_model, EXPERIMENT enum)
        (True, False, EXPERIMENT.EEG_CONTROL),
        (True, True, EXPERIMENT.EEG_AND_TEXT),
        (True, False, EXPERIMENT.EEG),
        (False, True, EXPERIMENT.TEXT),
        (True, True, EXPERIMENT.RANDOM_EEG_AND_TEXT),
        (True, False, EXPERIMENT.EEG_AND_RANDOM_TEXT),
    ]
    for use_eeg_model, use_lang_model, exp in variants:
        if args.experiment == exp:
            conf.set_table_config(
                modality=(
                    MODALITY.BIMODAL
                    if (use_eeg_model and use_lang_model)
                    else (MODALITY.EEG if use_eeg_model else MODALITY.TEXT)
                ),
                shuffle_attention_labels=(exp == EXPERIMENT.EEG_CONTROL),
                use_eeg_random_features=(
                    exp == EXPERIMENT.RANDOM_EEG_AND_TEXT
                ),
                use_text_random_features=(
                    exp == EXPERIMENT.EEG_AND_RANDOM_TEXT
                ),
                eeg_model=args.eeg_model,
                language_model=args.language_model,
            )
            run_table_experiment(
                conf,
                exp,
                args.eeg_model,
                args.language_model,
                use_eeg_model,
                use_lang_model,
            )


if __name__ == "__main__":
    main()
