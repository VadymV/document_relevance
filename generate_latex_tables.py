#!/usr/bin/env python3
"""
Script to analyze EEG/text model predictions for all EEG models.

Reads pickle files named <experiment>#language_model-<lm>#eeg_model-<em>.pkl,
computes ROC-AUC metrics (attention & semantic salience), aggregates by experiment_id, seed, and user,
runs permutation tests comparing each experiment against its control (including EEG-only and random-text conditions),
and collates summary statistics across EEG models.
"""
import argparse
import glob
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from doc_rel import EEG_MODEL, EXPERIMENT, parse_experiment
from doc_rel.configuration.configuration import Configuration
from doc_rel.misc.utils import set_logging
from doc_rel.model.embedding import LANG_MODEL
from doc_rel.misc.utils import run_permutation_test

logger = logging.getLogger(__name__)

# Helpers
display_names_lm = {
    'fasttext': 'FastText',
    'word2vec': 'Word2Vec',
    'glove': 'GloVe',
    'bert': 'BERT',
    'sbert': 'S-BERT',
}
display_names_em = {
    'eegnet': 'EEGNet',
    'lstm': 'LSTM',
    'lda': 'LDA',
    'svm': 'SVM',
}
# Display name mappings
em_order = ['eegnet', 'lstm', 'lda', 'svm']
lm_order = ['fasttext', 'word2vec', 'glove', 'bert', 'sbert']


# Helper formatters
def fmt_acc(m, s, bold: bool = False):
    m_str = f"{m:.2f}".replace('0.', '.')
    s_str = f"{s:.2f}".replace('0.', '.')
    if bold:
        return fr"\textbf{{{m_str}}} $\pm$ {s_str}"
    else:
        return fr"{m_str} $\pm$ {s_str}"


def fmt_pct(pct, p):
    if abs(pct) < 1:
        pct_str = f"{pct:.2f}".replace('0.', '.')
    else:
        pct_str = f"{pct:.2f}"
    star = '$^{\\ast}$' if p < 0.05 else ''
    return f"{pct_str}\\%{star}"


def parse_filename(
    filepath: str,
) -> Tuple[Optional[EXPERIMENT], Optional[str], Optional[str]]:
    """
    Parse experiment enum, language_model, and eeg_model from a filename.
    """
    base = os.path.basename(filepath)
    pattern = (
        r'^(?P<exp>[^#]+)'
        r'(?:#language_model-(?P<lm>[^#]+))?'
        r'(?:#eeg_model-(?P<em>[^#]+))?'
        r'\.pkl$'
    )
    m = re.match(pattern, base)
    if not m:
        logger.error("Unexpected filename: %s", base)
        raise ValueError("Unexpected filename: %s", base)
    exp_str, lm, em = m.group('exp'), m.group('lm'), m.group('em')
    try:
        exp_enum = parse_experiment(exp_str)
        logging.info(
            "Reading file %s. EM = %s; LM = %s, EXP = %s",
            base,
            em,
            lm,
            exp_str,
        )
    except ValueError:
        logger.error("Unknown experiment '%s'", exp_str)
        raise ValueError("Unknown experiment '%s'", exp_str)
    return exp_enum, lm, em


def read_results_files(
    dir_path: str, eeg_model_name: str, average_per_task: bool = True
) -> pd.DataFrame:
    """
    Load pickles for a specific EEG model and adds an 'exp_id'.
    """
    paths = glob.glob(os.path.join(dir_path, '*.pkl'))
    frames = []
    for p in paths:
        exp_enum, lm, em = parse_filename(p)
        if em and em != eeg_model_name:
            continue
        df = pd.read_pickle(p).copy()
        # build experiment identifier including language model
        exp_val = exp_enum.value if exp_enum else 'unknown'
        exp_id = f"{exp_val}#language_model-{lm}" if lm else exp_val
        df['exp_id'] = exp_id
        df['eeg_model'] = em or 'empty'
        df['lm_model'] = lm or 'empty'
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No results for EEG model {eeg_model_name}")

    df = pd.concat(frames, ignore_index=True)
    if len(df) != 13 * 600 + 5 * 600:
        raise ValueError("Mismatch of parsed and expected results")

    # Compute AUC attention (exclude pure-text)
    df['auc_attention'] = np.nan
    mask_att = ~df['eeg_model'].str.fullmatch('empty')
    df.loc[mask_att, 'auc_attention'] = df.loc[mask_att].apply(
        lambda r: roc_auc_score(r.attention_labels, r.predicted_attention),
        axis=1,
    )

    # Compute AUC semantic salience
    df['auc_salience'] = np.nan
    mask_sem = ~df['lm_model'].str.fullmatch('empty')
    df.loc[mask_sem, 'auc_salience'] = df.loc[mask_sem].apply(
        lambda r: roc_auc_score(
            r.semantic_salience_labels, r.predicted_semantic_salience
        ),
        axis=1,
    )

    df = df[
        [
            'user',
            'seed',
            'auc_attention',
            'auc_salience',
            'predicted_topical_document_relevance',
            'delta',
            'exp_id',
            'eeg_model',
            'lm_model',
        ]
    ]

    if average_per_task:
        #  Average over reading tasks
        df = (
            df.groupby(['user', 'seed', 'exp_id', 'eeg_model', 'lm_model'])
            .mean()
            .reset_index()
        )

    return df


def calculate_percentage_change(old: float, new: float) -> float:
    """
    Percentage change from old to new.
    """
    return 0.0 if old == 0 else (new - old) / abs(old) * 100.0


def process_model(
    df: pd.DataFrame, exp_ids_to_control: Dict[str, str]
) -> pd.DataFrame:
    """
    Compute metrics, aggregate, and run permutation tests for one EEG model.

    Returns summary DataFrame with columns:
      ['exp_id','seed','user',
       'auc_attention_mean','auc_attention_std',
       'auc_salience_mean','auc_salience_std',
       'predicted_topical_document_relevance_mean','predicted_topical_document_relevance_std',
       'pct_change','p_value']
    """

    summary = df.groupby(['exp_id'], as_index=False).agg(
        auc_attention_mean=('auc_attention', 'mean'),
        auc_attention_std=('auc_attention', 'std'),
        auc_salience_mean=('auc_salience', 'mean'),
        auc_salience_std=('auc_salience', 'std'),
        predicted_topical_document_relevance_mean=(
            'predicted_topical_document_relevance',
            'mean',
        ),
        predicted_topical_document_relevance_std=(
            'predicted_topical_document_relevance',
            'std',
        ),
    )

    # Precompute pct and p for each exp-control pair
    stats_map = {}
    for exp in summary['exp_id'].unique():
        ctrls = exp_ids_to_control.get(exp)
        if not ctrls:
            continue
        # ensure list
        ctrl_list = ctrls if isinstance(ctrls, list) else [ctrls]
        test_vals = df.loc[
            df['exp_id'] == exp,
            'predicted_topical_document_relevance',
        ]
        for ctrl in ctrl_list:
            ctrl_vals = df.loc[
                df['exp_id'] == ctrl,
                'predicted_topical_document_relevance',
            ]
            pct = calculate_percentage_change(
                ctrl_vals.mean(), test_vals.mean()
            )
            p = run_permutation_test(
                ctrl_vals, test_vals, alternative="less"
            )
            stats_map[(exp, ctrl)] = (pct, p)

    # Build expanded result: one row per summary row and per control
    records = []
    for _, row in summary.iterrows():
        exp = row['exp_id']
        ctrls = exp_ids_to_control.get(exp)
        if not ctrls:
            rec = row.to_dict()
            rec['control_id'] = None
            rec['pct_change'] = np.nan
            rec['p_value'] = np.nan
            records.append(rec)
        else:
            ctrl_list = ctrls if isinstance(ctrls, list) else [ctrls]
            for ctrl in ctrl_list:
                rec = row.to_dict()
                rec['control_id'] = ctrl
                pct, p = stats_map.get((exp, ctrl), (np.nan, np.nan))
                rec['pct_change'] = pct
                rec['p_value'] = p
                records.append(rec)
    result_df = pd.DataFrame.from_records(records)
    return result_df


def get_results(conf):

    # Build test-to-control mapping
    exp_ids_to_control: Dict[str, str] = {}
    for lm in LANG_MODEL:
        # control models
        eeg_text_id = (
            f"{EXPERIMENT.EEG_AND_TEXT.value}#language_model-{lm.value}"
        )
        # test models
        text_id = f"{EXPERIMENT.TEXT.value}#language_model-{lm.value}"
        random_eeg_text_id = f"{EXPERIMENT.RANDOM_EEG_AND_TEXT.value}#language_model-{lm.value}"
        eeg_random_text_id = f"{EXPERIMENT.EEG_AND_RANDOM_TEXT.value}"
        eeg_id = f"{EXPERIMENT.EEG.value}"

        # fill the dictionary
        exp_ids_to_control[text_id] = [eeg_text_id]
        exp_ids_to_control[random_eeg_text_id] = [eeg_text_id]
        exp_ids_to_control.setdefault(eeg_random_text_id, []).append(
            eeg_text_id
        )
        exp_ids_to_control.setdefault(eeg_id, []).append(eeg_text_id)

    all_summaries: List[pd.DataFrame] = []
    for em in EEG_MODEL:
        name = em.value
        logger.info("Processing EEG model: %s", name)
        df = read_results_files(
            os.path.join(conf.work_dir, args.pred_dir), name
        )
        summary = process_model(df, exp_ids_to_control)
        summary['eeg_model'] = name
        all_summaries.append(summary)

    final_df = pd.concat(all_summaries, ignore_index=True)
    out_path = os.path.join(conf.work_dir, 'summary_all_eeg_models.csv')
    final_df.to_csv(out_path, index=False)
    logger.info("Saved summary for all EEG models to %s", out_path)
    return final_df


def generate_bimodal_and_ablated_tables(final_df: pd.DataFrame) -> str:
    """
    Generate LaTeX code for the document relevance and ablation study tables.
    """
    # First table: full multimodal performance
    # Filter for EEG+TEXT runs

    df_copy = final_df.copy()
    df = df_copy[
        df_copy['exp_id'].str.startswith(
            f"{EXPERIMENT.EEG_AND_TEXT.value}#language_model-"
        )
    ]
    # Aggregate across seeds and participants
    table1 = {}
    for eeg in em_order:
        table1[eeg] = {}
        sub = df[df['eeg_model'] == eeg]
        for lm in lm_order:
            exp_id = f"{EXPERIMENT.EEG_AND_TEXT.value}#language_model-{lm}"
            m = sub[sub['exp_id'] == exp_id][
                'predicted_topical_document_relevance_mean'
            ]
            s = sub[sub['exp_id'] == exp_id][
                'predicted_topical_document_relevance_std'
            ]
            table1[eeg][lm] = (m.item(), s.item())
    # Find best overall
    best = max((v[0] for row in table1.values() for v in row.values()))
    # Build LaTeX
    latex = []
    latex.append(r"\begin{table}[hbp!]")
    latex.append(r"\normalsize\setlength{\tabcolsep}{3pt}")
    latex.append(r"\begin{minipage}[t]{.3\linewidth}")
    latex.append(
        r"\caption{Document relevance prediction results averaged over all participants and five runs with random seeding. SD: standard deviation. The best score across all human attention models and word embedding models is in bold font.}"
    )
    latex.append(r"\label{tab:document_relevance_results}")
    latex.append(r"\centering")
    latex.append(r"  \begin{tabular}{cccc}")
    latex.append(r"    \toprule")
    latex.append(
        r"        {\thead{Human \\ attention \\ model}} & {\thead{Embedding \\ model}} & {\thead{Accuracy $\pm$ 1 SD}} \\"
    )
    latex.append("         \midrule")
    for eeg in em_order:
        rows = []
        for lm in lm_order:
            m, s = table1[eeg][lm]
            bold = False
            if abs(np.round(m, 2) - np.round(best, 2)) < 1e-8:
                bold = True
            cell = fmt_acc(m, s, bold)
            rows.append((display_names_lm[lm], cell))
        latex.append(
            fr"     \multirow{{5}}{{*}}{{{display_names_em[eeg]}}} & {rows[0][0]} & {rows[0][1]} \\"
        )
        for name, cell in rows[1:]:
            latex.append(fr"     & {name} & {cell} \\")
        latex.append(r"     \midrule")
    latex[-1] = latex[-1].replace('\midrule', r'\bottomrule')
    latex.append(r"\end{tabular}")
    latex.append(r"\end{minipage}\hfill")
    # Second block: Ablation text modality (EEG only)
    latex.append(r"\begin{minipage}[t]{.67\linewidth}")
    latex.append(
        r"\caption{Ablation study: prediction of document relevance using (a) EEG and (b) text as the only input. SD: standard deviation. $^{\ast}$ denotes statistical difference to the models in Table~\ref{tab:document_relevance_results} with $p$-value $<0.05$. $\dagger$ Change in performance against the models in Table~\ref{tab:document_relevance_results} measured in \% using FastText, GloVe, Word2Vec, BERT, and S-BERT as word embedding models. The best score is in bold font.}\label{tab:ablation_study}"
    )
    latex.append(
        r"\subcaption{The text modality is ablated \\ (the EEG modality is the only input). \label{tab:ablation_text}}"
    )
    latex.append(r"\centering")
    latex.append(r"  \begin{tabular}{ccccccc}")
    latex.append(r"    \toprule")
    latex.append(
        r"        \multirow{3}{*}{\thead{Human \\ attention \\ model}} &\multirow{3}{*}{\thead{Accuracy $\pm$ 1 SD}} &\multicolumn{5}{c}{\thead{Change$^\dagger$}} \\"
    )
    latex.append(r"        \cmidrule(rl){3-7}")
    latex.append(
        r"        & & \thead{FastText} & \thead{Word2Vec} & \thead{GloVe} & \thead{BERT} & \thead{S-BERT}\\"
    )
    latex.append(r"         \midrule")
    # Prepare ablation text data
    df_eeg = df_copy[df_copy['exp_id'] == EXPERIMENT.EEG.value]
    # Find best overall
    best = df_eeg["predicted_topical_document_relevance_mean"].max()
    for eeg in em_order:
        sub = df_eeg[df_eeg['eeg_model'] == eeg]
        m = sub[
            'predicted_topical_document_relevance_mean'
        ].mean()  #  the same values are repeated
        s = sub[
            'predicted_topical_document_relevance_std'
        ].mean()  #  the same values are repeated
        bold = False
        if abs(np.round(m, 2) - np.round(best, 2)) < 1e-8:
            bold = True
        row = fmt_acc(m, s, bold)
        changes = []
        for lm in lm_order:
            control_id = (
                f"{EXPERIMENT.EEG_AND_TEXT.value}#language_model-{lm}"
            )
            # Find pct and p in df
            r = df_copy[
                (df_copy['exp_id'] == EXPERIMENT.EEG.value)
                & (df_copy['control_id'] == control_id)
                & (df_copy['eeg_model'] == eeg)
            ]
            pct = r['pct_change'].iloc[0]
            p = r['p_value'].iloc[0]
            changes.append(fmt_pct(pct, p))
        latex.append(
            fr"     {display_names_em[eeg]} & {row} & {' & '.join(changes)} \\"
        )
    latex.append(r"  \bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\centering")
    latex.append(
        r"\subcaption{The EEG modality is ablated \\ (the text modality is the only input).}\label{tab:ablation_eeg}"
    )
    latex.append(r"\begin{tabular}{cccccc}")
    latex.append(r"\toprule")
    latex.append(
        r"    \multirow{3}{*}{\thead{Embedding \\ model}} &\multirow{3}{*}{\thead{Accuracy $\pm$ 1 SD}} &\multicolumn{3}{c}{\thead{Change$^\dagger$}} \\"
    )
    latex.append(r"    \cmidrule(rl){3-6}")
    latex.append(
        r"    & & \thead{EEGNet} & \thead{LSTM} & \thead{LDA} & \thead{SVM}\\"
    )
    latex.append("     \midrule")
    # Prepare ablation eeg data
    df_text = df_copy[
        df_copy['exp_id'].str.startswith(
            f"{EXPERIMENT.TEXT.value}#language_model-"
        )
    ]
    best = df_text["predicted_topical_document_relevance_mean"].max()
    for lm in lm_order:
        sub = df_text[
            df_text['exp_id']
            == f"{EXPERIMENT.TEXT.value}#language_model-{lm}"
        ]
        m = sub[
            'predicted_topical_document_relevance_mean'
        ].mean()  #  the same values are repeated
        s = sub[
            'predicted_topical_document_relevance_std'
        ].mean()  #  the same values are repeated
        bold = False
        if abs(np.round(m, 2) - np.round(best, 2)) < 1e-8:
            bold = True
        val = fmt_acc(m, s, bold)
        changes = []
        for eeg in em_order:
            test_id = f"{EXPERIMENT.TEXT.value}#language_model-{lm}"
            control_id = (
                f"{EXPERIMENT.EEG_AND_TEXT.value}#language_model-{lm}"
            )
            r = df_copy[
                (df_copy['exp_id'] == test_id)
                & (df_copy['control_id'] == control_id)
                & (df_copy['eeg_model'] == eeg)
            ]
            pct = r['pct_change'].iloc[0]
            p = r['p_value'].iloc[0]
            changes.append(fmt_pct(pct, p))
        change_str = ' & '.join(changes)
        lm_disp = display_names_lm[lm]
        latex.append(fr"    {lm_disp} & {val} & {change_str} \\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{minipage}")
    latex.append(r"\end{table}")
    return '\n'.join(latex)


def generate_human_attention_table(final_df: pd.DataFrame) -> str:
    """
    Generate LaTeX table for human attention and semantic salience AUROC results.
    """

    df = final_df.copy()
    df = df[
        df["exp_id"].isin(
            [EXPERIMENT.EEG.value, EXPERIMENT.EEG_CONTROL.value]
        )
    ]
    df = (
        df[
            [
                "exp_id",
                "auc_attention_mean",
                "auc_attention_std",
                "eeg_model",
            ]
        ]
        .drop_duplicates()
        .set_index("eeg_model")
    )

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\normalsize\setlength{\tabcolsep}{10pt}")
    lines.append(
        r"\caption{Prediction results of the human attention models averaged over all participants and five runs with random seeding. For the control model, random permutations of the labels are used. SD: standard deviation.}"
    )
    lines.append(r"\label{tab:results_human_attention}")
    lines.append(r"  \begin{tabular}{ccccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"        \multirow{3}{*}{\thead{Model}} & \multicolumn{4}{c}{\thead{AUROC $\pm$ 1 SD}} \\"
    )
    lines.append(r"        \cmidrule(rl){2-5}")
    lines.append(
        r"         & \thead{EEGNet} & \thead{LSTM} & \thead{LDA} & \thead{SVM} \\"
    )
    lines.append(r"         \midrule")

    # Permuted human attention control model row
    df_row1 = df[df["exp_id"].isin([EXPERIMENT.EEG_CONTROL.value])]
    vals = [
        f"{df_row1.loc[em,'auc_attention_mean']:.2f} $\\pm$ {df_row1.loc[em,'auc_attention_std']:.2f}"
        for em in em_order
    ]
    lines.append(
        r"     \makecell{Permuted human attention control model} & "
        + " & ".join(vals)
        + r" \\"
    )

    # Human attention row
    df_row2 = df[df["exp_id"].isin([EXPERIMENT.EEG.value])]
    vals = [
        f"{df_row2.loc[em,'auc_attention_mean']:.2f} $\\pm$ {df_row2.loc[em,'auc_attention_std']:.2f}"
        for em in em_order
    ]
    lines.append(
        r"     Human attention model & " + " & ".join(vals) + r" \\"
    )

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_gating_tables(final_df: pd.DataFrame) -> str:
    """
    Generate LaTeX code for the gating study tables using final_df.
    """
    df_copy = final_df.copy()
    lines = []
    lines.append(r"\begin{table}[!htb]")
    lines.append(
        r"\caption{Document relevance results, where (a) the text modality is gated (random input instead of word embeddings is provided) and (b) the EEG modality is gated (random input instead of brain recordings is provided). SD: standard deviation. $^{\ast}$ denotes statistical difference to the models in Table~\ref{tab:document_relevance_results} with $p$-value $<0.05$. $\dagger$ Change in performance against the corresponding models in Table~\ref{tab:document_relevance_results} measured in \% . The best score is in bold font.}"
    )
    lines.append(r"\label{tab:gating_study}")
    lines.append(r"\normalsize\setlength{\tabcolsep}{3pt}")

    # (a) text gated
    lines.append(r"\begin{minipage}[t]{.59\linewidth}\centering")
    lines.append(
        r"\subcaption{The text modality is gated.}\label{tab:gating_text}"
    )
    lines.append(r"\begin{tabular}{ccccccc}")
    lines += [
        r"    \toprule",
        r"        \multirow{3}{*}{\thead{Human \\ attention \\ model}} &",
        r"        \multirow{3}{*}{\thead{Accuracy \\ $\pm$ 1 SD}} &",
        r"        \multicolumn{4}{c}{\thead{Change$^\dagger$}} \\",
        r"        \cmidrule(rl){3-7}",
        r"        & & \thead{FastText} & \thead{Word2Vec} & \thead{GloVe} & \thead{BERT} & \thead{S-BERT}\\",
        r"         \midrule",
    ]
    df_text = df_copy[df_copy['exp_id'] == 'eeg+random_text']
    best = df_text['predicted_topical_document_relevance_mean'].max()
    for em in em_order:
        sub = df_text[df_text['eeg_model'] == em]
        m = sub[
            'predicted_topical_document_relevance_mean'
        ].mean()  #  the same values are repeated
        s = sub[
            'predicted_topical_document_relevance_std'
        ].mean()  #  the same values are repeated
        bold = False
        if abs(np.round(m, 2) - np.round(best, 2)) < 1e-8:
            bold = True
        changes = []
        for lm in lm_order:
            r = sub[sub['control_id'].str.contains(lm)]
            if not r.empty:
                pct = r['pct_change'].iloc[0]
                pval = r['p_value'].iloc[0]
                changes.append(fmt_pct(pct, pval))
            else:
                raise ValueError("Control experiment is not found!")
        line = (
            fr"     {display_names_em[em]} & {fmt_acc(m, s, bold)} & "
            + " & ".join(changes)
            + r" \\"
        )
        lines.append(line)
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{minipage}")

    # (b) EEG gated
    lines.append(r"\begin{minipage}[t]{.4\linewidth}\centering")
    lines.append(
        r"\subcaption{The EEG modality is gated.}\label{tab:gating_eeg}"
    )
    lines.append(r"\begin{tabular}{cccc}")
    lines += [
        r"    \toprule",
        r"    \thead{Human \\ attention \\ model} & \thead{Embedding \\ model} & \thead{Accuracy \\ $\pm$ 1 SD} & \thead{Change$^\dagger$}\\",
        r"    \midrule",
    ]
    df_eeg = df_copy[df_copy['exp_id'].str.contains('random_eeg')]
    best = df_eeg['predicted_topical_document_relevance_mean'].max()
    for em in em_order:
        for i, lm in enumerate(lm_order):
            r = df_eeg[
                (df_eeg['eeg_model'] == em)
                & (df_eeg['control_id'].str.contains(lm))
            ]
            if not r.empty:
                m = r['predicted_topical_document_relevance_mean'].iloc[0]
                s = r['predicted_topical_document_relevance_std'].iloc[0]
                pct, pval = r['pct_change'].iloc[0], r['p_value'].iloc[0]
                bold = False
                if abs(np.round(m, 2) - np.round(best, 2)) < 1e-8:
                    bold = True
                acc_str = fmt_acc(m, s, bold)
                change_str = fmt_pct(pct, pval)
            else:
                raise ValueError("Control experiment is not found!")
            prefix = (
                f"     \\multirow{{{len(lm_order)}}}{{*}}{{{display_names_em[em]}}} & "
                if i == 0
                else "     & "
            )
            lines.append(
                prefix
                + f"{display_names_lm[lm]} & {acc_str} & {change_str} \\\\"
            )
        lines.append(r"     \midrule")
    lines[-1] = lines[-1].replace('\midrule', r'\bottomrule')
    lines.append(r"\end{tabular}")
    lines.append(r"\end{minipage}\hfill")
    lines.append(r"\begin{minipage}[t]{.5\linewidth}\centering")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', default='predictions')
    args = parser.parse_args()

    conf = Configuration()
    set_logging(conf.work_dir, "results")
    results = get_results(conf)
    # results = pd.read_csv(
    # os.path.join(conf.work_dir, 'summary_all_eeg_models.csv')
    # )

    # Generate LaTeX tables
    print(generate_human_attention_table(results))
    print("#" * 20)
    print(generate_bimodal_and_ablated_tables(results))
    print("#" * 20)
    print(generate_gating_tables(results))
