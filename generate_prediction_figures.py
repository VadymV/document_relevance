import os

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from doc_rel.configuration.configuration import Configuration
from generate_latex_tables import (
    read_results_files,
    display_names_em,
    display_names_lm,
    em_order,
    lm_order,
)
from doc_rel.vis.model_performance import (
    plot_modality_scatterplot,
    plot_boxplot,
    plot_correlation,
)

from doc_rel.misc.utils import (
    set_seed,
)

from doc_rel import EXPERIMENT
from typing import Literal


def create_figure_five():
    # --- Generate Figure 5 in the paper---:
    sns.set_theme(font_scale=2)
    a4_dims = (25, 5)
    fig, axes = plt.subplots(figsize=a4_dims, ncols=4, sharey=True)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    for axis_id, em in enumerate(em_order):
        if axis_id == 0:
            title = display_names_em[em]
        elif axis_id == 1:
            title = display_names_em[em]
        elif axis_id == 2:
            title = display_names_em[em]
        elif axis_id == 3:
            title = display_names_em[em]
        results = read_results_files(
            os.path.join(conf.work_dir, "predictions"),
            em,
            average_per_task=False,
        )
        results["user"] = results["user"].map(user_mapping)
        results["exp_id"] = results["exp_id"].str.split('#').str[0]
        columns_of_topical_relevance = [
            "auc_attention",
            "auc_salience",
            "user",
            "seed",
            "exp_id",
        ]

        plot_data = results[columns_of_topical_relevance]
        plot_data = plot_data[
            plot_data["exp_id"].isin([EXPERIMENT.EEG.value])
        ]
        control_data = results[
            results["exp_id"].isin([EXPERIMENT.EEG_CONTROL.value])
        ]

        order = (
            plot_data[["user", "auc_attention"]]
            .groupby(by=["user"], as_index=False)
            .mean(numeric_only=True)
            .sort_values("auc_attention", ascending=False)
            .user.tolist()
        )
        plot_boxplot(
            data=plot_data,
            x="user",
            y="auc_attention",
            work_dir=conf.work_dir,
            file_id=f"human_attention_{em}",
            x_label="",
            y_label="AUROC score",
            title=title,
            average=plot_data[["auc_attention"]].mean().item(),
            average2=None,
            random_baseline=control_data[["auc_attention"]].mean().item(),
            order=order,
            ax=axes[axis_id],
        )

        axes[axis_id].label_outer()

    fig.supxlabel(
        "Participant",
        fontsize=28,
        y=-0.15,
    )
    plt.ylim((0.25, 0.85))
    plt.xticks(rotation=45)
    plt.savefig(
        "{}/plots/{}.pdf".format(
            conf.work_dir, "predicted_attention_per_participant"
        ),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def create_figure_six():
    sns.set_theme(font_scale=2.2)
    # --- Generate Figure 6 in the paper---:
    for lm in lm_order:
        a4_dims = (26, 10)
        fig, axes = plt.subplots(
            figsize=a4_dims,
            ncols=len(em_order),
            nrows=2,
            sharey=True,
            sharex=True,
        )
        axes = axes.reshape(
            8,
        )
        plt.subplots_adjust(wspace=0.03, hspace=0.03, bottom=0.11, left=0.10)
        x_min, y_min = 1, 1
        x_max, y_max = 0, 0
        for axis_id, em in enumerate(em_order * 2):
            columns = [
                "predicted_topical_document_relevance",
                "user",
                "seed",
                "exp_id",
                "lm_model",
                "eeg_model",
            ]

            plot_data = results_cache[em][columns]
            plot_data["exp_id"] = plot_data["exp_id"].str.split('#').str[0]
            plot_data = plot_data[
                plot_data["exp_id"].isin(
                    [
                        EXPERIMENT.EEG_AND_TEXT.value,
                        EXPERIMENT.EEG.value,
                        EXPERIMENT.TEXT.value,
                    ]
                )
            ]

            plot_data = (
                plot_data.groupby(
                    by=["user", "exp_id", "lm_model", "eeg_model"],
                    as_index=True,
                )
                .mean(numeric_only=True)[
                    ["predicted_topical_document_relevance"]
                ]
                .reset_index()
                .sort_values(by=["user"])
            )

            # Bi-modal input:
            x = plot_data[
                (plot_data["exp_id"] == EXPERIMENT.EEG_AND_TEXT.value)
                & (plot_data["lm_model"] == lm)
                & (plot_data["eeg_model"] == em)
            ]["predicted_topical_document_relevance"].tolist()

            # Uni-modal input:
            if axis_id <= 3:
                y = plot_data[
                    (plot_data["exp_id"] == EXPERIMENT.EEG.value)
                    & (plot_data["eeg_model"] == em)
                ]["predicted_topical_document_relevance"].tolist()
            else:
                y = plot_data[
                    (plot_data["exp_id"] == EXPERIMENT.TEXT.value)
                    & (plot_data["lm_model"] == lm)
                ]["predicted_topical_document_relevance"].tolist()

            annotations = plot_data["user"].astype(int).unique()

            plot_modality_scatterplot(
                x=x,
                y=y,
                file_id=f"document_relevance__{lm}_vs_unimodal",
                x_label="",
                y_label="EEG" if axis_id <= 3 else display_names_lm[lm],
                title=display_names_em[em] if axis_id <= 3 else None,
                annotations=annotations,
                work_dir=conf.work_dir,
                ax=axes[axis_id],
                fontsize=18,
            )

            if np.min(x) < x_min:
                x_min = np.min(x)
            if np.min(y) < y_min:
                y_min = np.min(y)
            if np.max(x) > x_max:
                x_max = np.max(x)
            if np.max(y) > y_max:
                y_max = np.max(y)

        offset = 0.03
        plt.xlim((x_min - offset, x_max + offset))
        plt.ylim(y_min - offset, y_max + offset)
        fig.supxlabel(
            f"Accuracy score for a bimodal input (EEG + {display_names_lm[lm]})"
        )
        fig.supylabel("Accuracy score for a unimodal input")
        plt.savefig(
            "{}/plots/predicted_{}-{}.pdf".format(
                conf.work_dir,
                "document_relevance",
                lm,
            ),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()


def create_supp_figure_relationship():
    sns.set_theme(font_scale=1.6)
    ncols = len(em_order)
    nrows = len(lm_order)
    fig, axes = plt.subplots(
        figsize=(width_per_subplot * ncols, height_per_subplot * nrows),
        ncols=ncols,
        nrows=nrows,
        sharey=True,
        sharex=True,
        gridspec_kw=gridspec_kw,
    )
    axes = axes.flatten()

    x_min, y_min, x_max, y_max = 1, 1, 0, 0
    global_axis_id = 0

    for lm in lm_order:
        for em in em_order:
            df = results_cache[em][
                [
                    "user",
                    "auc_attention",
                    "predicted_topical_document_relevance",
                    "exp_id",
                    "lm_model",
                    "eeg_model",
                ]
            ]

            filtered = df[
                df["exp_id"].isin(
                    [EXPERIMENT.EEG_AND_TEXT.value, EXPERIMENT.EEG.value]
                )
            ]

            plot_data = (
                filtered.groupby(
                    ["user", "exp_id", "lm_model", "eeg_model"],
                    as_index=False,
                )
                .mean(numeric_only=True)
                .sort_values(by="user")
            )

            x = plot_data[
                (plot_data["exp_id"] == EXPERIMENT.EEG_AND_TEXT.value)
                & (plot_data["lm_model"] == lm)
                & (plot_data["eeg_model"] == em)
            ]["predicted_topical_document_relevance"].tolist()

            y = plot_data[
                (plot_data["exp_id"] == EXPERIMENT.EEG.value)
                & (plot_data["eeg_model"] == em)
            ]["auc_attention"].tolist()

            plot_modality_scatterplot(
                x=x,
                y=y,
                file_id="human_attention_vs_document_relevance",
                x_label="",
                y_label="",
                title=(
                    display_names_em[em]
                    if global_axis_id < len(em_order)
                    else None
                ),
                annotations=plot_data["user"].astype(int).unique(),
                work_dir=conf.work_dir,
                ax=axes[global_axis_id],
            )
            x_min, y_min = min(x_min, np.min(x)), min(y_min, np.min(y))
            x_max, y_max = max(x_max, np.max(x)), max(y_max, np.max(y))
            global_axis_id += 1

    for i, ax in enumerate(axes):
        if (i + 1) % len(em_order) == 0:
            lm_idx = i // len(em_order)
            ax.text(
                x_max + 0.03,
                (y_min + y_max) / 2,
                display_names_lm[lm_order[lm_idx]],
                fontsize=20,
                va="center",
                ha="left",
                rotation=-90,
                transform=ax.transData,
            )

    fig.supxlabel("Predicted document relevance", fontsize=20)
    fig.supylabel("Predicted human attention", fontsize=20)
    fig.subplots_adjust(left=0.1, bottom=0.06)
    lim_offset = 0.03
    plt.xlim((x_min - lim_offset, x_max + lim_offset))
    plt.ylim((y_min - lim_offset, y_max + lim_offset))
    plt.savefig(
        os.path.join(
            conf.work_dir,
            "plots",
            "human_attention_vs_document_relevance.pdf",
        ),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def _create_fraction_figure(
    results: pd.DataFrame,
    eeg_model: str,
):
    plot_data_dict = {}
    for language_model in lm_order:

        #  predictions per word
        fname = f"eeg+text#language_model-{language_model}#eeg_model-{eeg_model}.csv"
        words_df = pd.read_csv(os.path.join(predictions_folder, fname))

        results_df = results.copy()
        results_df["exp_id"] = results_df["exp_id"].str.split('#').str[0]
        results_df = results_df[
            (results_df["exp_id"] == EXPERIMENT.EEG_AND_TEXT.value)
            & (results_df["lm_model"] == language_model)
            & (results_df["eeg_model"] == eeg_model)
        ]
        if len(results_df) != 600:
            raise ValueError("There are must be 600 rows.")
        results_df["fold"] = (results_df.index % 8) + 1
        results_df = results_df[
            ["user", "fold", "seed", "predicted_topical_document_relevance"]
        ]
        df = words_df.merge(
            results_df, how='left', on=["user", "fold", "seed"]
        )

        def calculate_fraction(
            data: pd.DataFrame,
            document: Literal["relevant", "irrelevant"],
            language_model: str,
        ) -> pd.DataFrame:
            rows = []
            for (user, fold, seed), grp in data.groupby(
                ["user", "fold", "seed"]
            ):
                # document mask
                doc = grp[
                    (
                        grp["relevant_document"] == True
                        if document == "relevant"
                        else grp["relevant_document"] == False
                    )
                ]
                # fraction of semantically salient words
                frac = doc["semantic_relevance"].mean()
                # get predicted document relevance
                pred = doc["predicted_topical_document_relevance"].mean()
                rows.append((frac, pred))

            df = pd.DataFrame(rows, columns=["fraction", "prediction"])
            # bin into 100 bins
            bins = np.linspace(0, 1, 101)
            df["fraction_bin"] = pd.cut(
                df["fraction"], bins=bins, right=False, precision=3
            )
            agg = (
                df.groupby("fraction_bin", observed=True)["prediction"]
                .mean()
                .reset_index()
            )
            agg["bin"] = agg.index / 100 + 1 / 200  # midpoint
            return agg[agg["prediction"].notna()]

        # compute for relevant / irrelevant
        data_rel = calculate_fraction(df, "relevant", language_model)
        data_irrel = calculate_fraction(df, "irrelevant", language_model)

        plot_data_dict[language_model] = (data_rel, data_irrel)

    # plot side-by-side
    sns.set(font_scale=1)
    fig, axes = plt.subplots(
        len(lm_order),
        2,
        figsize=(width_per_subplot * 2, height_per_subplot * len(lm_order)),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.03, wspace=0.03)
    titles = ["Relevant documents", "Irrelevant documents"]
    lm_ids = [x for x in range(len(lm_order)) for _ in range(2)]
    x_min = -0.03
    x_max = 0.33
    y_min = -0.03
    y_max = 1.03
    for axis_id, ax in enumerate(axes):
        col_id = axis_id % 2
        selected_lm = lm_order[lm_ids[axis_id]]
        d = plot_data_dict[selected_lm][col_id]
        plot_correlation(
            ax=ax,
            input_x=d["bin"].to_numpy(),
            response_y=d["prediction"].to_numpy(),
            annotations=None,
            groups=None,
            is_legend_visible=False,
            with_text_annotations=None,
            y_lim=[y_min, y_max],
            x_lim=[x_min, x_max],
            text_position=(0.1, 0.1),
            title=titles[col_id] if axis_id < 2 else None,
            y_label="",
            x_label="",
            fontsize=10,
            s=100,
        )

        if col_id == 1:
            ax.text(
                x_max + 0.01,
                (y_min + y_max) / 2,
                display_names_lm[selected_lm],
                fontsize=14,
                va="center",
                ha="left",
                rotation=-90,
                transform=ax.transData,
            )

    fig.supxlabel("Fraction of semantically salient words", fontsize=14)
    fig.supylabel("Fraction of correctly classified documents", fontsize=14)
    fig.subplots_adjust(left=0.1, bottom=0.06)

    out_path = os.path.join(
        conf.work_dir,
        "plots",
        f"salience_vs_doc_rel_{eeg_model}.pdf",
    )
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def create_supp_figure_fraction():
    for em in em_order:
        results = read_results_files(predictions_folder, em, False)
        _create_fraction_figure(results, em)


if __name__ == "__main__":
    conf = Configuration()
    conf.selected_seed = 1
    set_seed(conf.selected_seed)
    gridspec_kw = {"hspace": 0.01, "wspace": 0.01}
    width_per_subplot = 4
    height_per_subplot = 4

    user_mapping = {
        "TRPB101": 1,
        "TRPB102": 2,
        "TRPB103": 3,
        "TRPB105": 4,
        "TRPB106": 5,
        "TRPB107": 6,
        "TRPB109": 7,
        "TRPB110": 8,
        "TRPB111": 9,
        "TRPB112": 10,
        "TRPB113": 11,
        "TRPB114": 12,
        "TRPB115": 13,
        "TRPB116": 14,
        "TRPB117": 15,
    }

    # Preload all result files once per EEG model
    predictions_folder = os.path.join(conf.work_dir, "predictions")
    results_cache = {
        em: read_results_files(predictions_folder, em) for em in em_order
    }

    # Map users to an int value and set exp_id without language model
    for em, df in results_cache.items():
        df["user"] = df["user"].map(user_mapping)
        df["exp_id"] = df["exp_id"].str.split('#').str[0]

    create_figure_five()
    create_figure_six()
    create_supp_figure_fraction()
    create_supp_figure_relationship()
