import copy
import glob
import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt, transforms
from matplotlib.ticker import FormatStrFormatter

from doc_rel.misc.utils import calculate_kendall_tau

sns.set(font_scale=2.2)


def plot_all_results(ckpt_dir, nth_tick, epochs):
    all_files = glob.glob(os.path.join(ckpt_dir, "metrics_*.pkl"))
    data = pd.concat(
        (pd.read_pickle(f) for f in all_files), ignore_index=True
    )
    # Accuracy:
    plot_results(
        ckpt_dir,
        x=data["epoch"].tolist(),
        y=data["train_acc"].tolist(),
        hue=data["user"].tolist(),
        y_label="Training accuracy",
        file_name_suffix="all_results_acc",
        nth_tick=nth_tick,
        y_lim=[0.4, 1.01],
    )
    plot_results(
        ckpt_dir,
        x=data["epoch"].tolist(),
        y=data["test_acc"].tolist(),
        hue=data["user"].tolist(),
        y_label="Test accuracy",
        file_name_suffix="all_results_acc",
        nth_tick=nth_tick,
        y_lim=[0.4, 1.01],
    )
    # MCC:
    plot_results(
        ckpt_dir,
        x=data["epoch"].tolist(),
        y=data["train_mcc"].tolist(),
        hue=data["user"].tolist(),
        y_label="Training MCC",
        file_name_suffix="all_results_mcc",
        nth_tick=nth_tick,
        y_lim=[-0.2, 1.01],
    )
    plot_results(
        ckpt_dir,
        x=data["epoch"].tolist(),
        y=data["test_mcc"].tolist(),
        hue=data["user"].tolist(),
        y_label="Test MCC",
        file_name_suffix="all_results_mcc",
        nth_tick=nth_tick,
        y_lim=[-0.2, 1.01],
    )


def plot_results(
    ckpt_dir,
    x,
    y,
    hue,
    y_label,
    file_name_suffix="",
    nth_tick=10,
    y_lim=None,
):
    if y_lim is None:
        y_lim = [0, 1.01]
    plt.figure(figsize=(16, 10))
    ax = sns.lineplot(x=x, y=y, hue=hue, errorbar="ci")

    ax.set_xlabel(xlabel="Epoch")
    ax.set_ylabel(ylabel=y_label)
    ax.set_xticks(range(1, len(set(x)) + 1))  # <--- set the ticks first
    ax.set_xticklabels(range(1, len(set(x)) + 1))
    ax.set_ylim(y_lim)
    ax.legend(loc="upper left")

    for ind, label in enumerate(ax.get_xticklabels()):
        if ind == 0 or (ind + 1) % nth_tick == 0:  # every nth label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    plt.title("Performance of the model.")
    plt.savefig(
        "{}/epoch_vs_loss_{}_data_{}.pdf".format(
            ckpt_dir, y_label, file_name_suffix
        ),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_delta(work_dir, data, file_id, title=None):
    a4_dims = (15, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    data["Prediction"] = np.where(
        data["test_doc_auc"] == 1, "Correct", "Incorrect"
    )
    order = ["Incorrect", "Correct"]
    ax = sns.boxplot(
        data=data, x="Prediction", y="delta", order=order, ax=ax
    )
    sns.stripplot(
        data=data[data["participant"].isin(["02", "05"])],
        x="Prediction",
        y="delta",
        hue="participant",
        size=10,
        palette="rocket",
        order=order,
        dodge=True,
    )
    sns.stripplot(
        data=data,
        x="Prediction",
        y="delta",
        hue="participant",
        color="black",
        alpha=0.3,
        size=10,
        order=order,
        legend=False,
    )
    ax.xaxis.grid(True)
    sns.despine(trim=True, left=True)
    # add_stat_annotation(ax, data=data, x="Prediction", y="delta", order=order,
    #                     box_pairs=[("Incorrect", "Correct")],
    #                     test='Kruskal', text_format='star', loc='outside', verbose=2)
    ax.set_xlabel(xlabel="Prediction")
    ax.set_ylabel(ylabel="Delta")
    plt.legend(title="Participant")
    plt.title(title)
    plt.savefig(
        "{}/plots/results_{}.pdf".format(work_dir, file_id),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_bar(
    work_dir,
    x,
    y,
    x_label,
    y_label,
    file_id,
    y_lim=None,
    title=None,
    average=None,
    hue=None,
    legend_title=None,
    rotate=True,
    palette=None,
    lines=None,
    random_baseline=None,
    order=None,
):
    a4_dims = (15, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.barplot(
        ax=ax, x=x, y=y, errorbar="se", hue=hue, palette=palette, order=order
    )

    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)

    if rotate:
        for item in ax.get_xticklabels():
            item.set_rotation(45)
    if y_lim is not None:
        plt.ylim(y_lim)
    if average is not None:
        plt.axhline(y=average, color="r", linestyle="-")
        trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData
        )
        ax.text(
            0,
            average,
            "{:.2f}".format(average),
            color="red",
            transform=trans,
            ha="right",
            va="center",
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Plot the random baseline:
    if random_baseline is not None:
        plt.axhline(y=random_baseline, color="blue", linestyle="--")
        trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData
        )
        ax.text(
            0,
            random_baseline,
            "{:.2f}".format(random_baseline),
            color="red",
            transform=trans,
            ha="right",
            va="center",
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if legend_title is not None:
        ax.legend(title=legend_title, loc="lower right")

    if lines is not None:
        ax2 = ax.twinx()
        ax2.grid(False)
        ax.set_ylabel(ylabel="")
        temp = {"user": x, "topical_relevance": lines}
        temp = pd.DataFrame(temp)
        topical_relevance = temp.groupby(by=["user"]).mean()[
            "topical_relevance"
        ]
        sns.lineplot(
            ax=ax2,
            data=topical_relevance,
            marker="o",
            sort=False,
            color="grey",
        )

    yticks = ax.yaxis.get_major_ticks()
    for i in range(len(yticks)):
        if i == 4 or i == 5:
            yticks[i].set_visible(False)

    plt.title(title)
    plt.savefig(
        "{}/plots/results_{}.pdf".format(work_dir, file_id),
        format="pdf",
        bbox_inches="tight",
    )

    plt.close()


def plot_boxplot(
    data,
    x,
    y,
    work_dir,
    file_id,
    x_label,
    y_label,
    y_lim=None,
    title=None,
    average=None,
    average2=None,
    random_baseline=None,
    order=None,
    ax=None,
    set_ticks=True,
    fontsize=None,
):
    if ax is None:
        a4_dims = (10, 7.27)
        fig, ax = plt.subplots(figsize=a4_dims)
    palette = sns.color_palette("Greens", n_colors=15)
    palette.reverse()
    sns.boxplot(
        ax=ax,
        data=data,
        x=x,
        y=y,
        order=order,
        palette=palette,
    )
    if x_label is not None:
        ax.set_xlabel(xlabel=x_label, labelpad=30)
    if y_label is not None:
        ax.set_ylabel(ylabel=y_label, labelpad=30)
    if average is not None:
        ax.axhline(y=average, color="r", linestyle="-")
    if average2 is not None:
        ax.axhline(y=average2, color="black", linestyle="dotted")
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Plot the random baseline:
    if random_baseline is not None:
        ax.axhline(y=random_baseline, color="blue", linestyle="--")

    if set_ticks:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        ax.set_yticklabels(np.arange(0.4, 1.01, 0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title(title)
    if ax is None:
        plt.xticks(rotation=45)
        plt.title(title)
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.savefig(
            "{}/plots/results_{}.pdf".format(work_dir, file_id),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_main_components(
    work_dir,
    file_id,
    x_label,
    y_label,
    epos_list,
    meta_data,
    title=None,
    y_lim=None,
):
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["font.size"] = "22"

    # # stop_words = set(stopwords.words('english'))
    meta_data["Relevant term"] = False
    meta_data.loc[
        meta_data["type"].isin(["id", "semrel", "semanti"]), "Relevant term"
    ] = True

    def extract_components(epos_list, meta_data, tmin, tmax, component):
        epos_list_copy = copy.deepcopy(epos_list)
        meta_data_copy = copy.deepcopy(meta_data)

        eeg_data = np.vstack(
            [
                epos_list_copy[i]
                .pick_channels(["Pz"])
                .crop(tmin=tmin, tmax=tmax)
                .get_data()
                .squeeze()
                for i in range(len(epos_list_copy))
            ]
        )
        eeg_data = np.mean(eeg_data, axis=1)

        meta_data_copy["Component"] = component
        meta_data_copy["Value"] = eeg_data
        meta_data_copy = meta_data_copy[
            meta_data_copy["Value"].abs() < 0.000003
        ]
        meta_data_copy = meta_data_copy[meta_data_copy["isrelsent"] == True]

        # words = set.intersection(set(meta_data_copy[meta_data_copy["isrelsent"] == True]["word"].tolist()),
        #             set(meta_data_copy[meta_data_copy["isrelsent"] == False]["word"].tolist()))
        #
        # meta_data_copy = meta_data_copy[meta_data_copy["word"].isin(words)]
        #
        # group1 = meta_data_copy[meta_data_copy["isrelsent"] == True].sample(frac=1).drop_duplicates(subset=['word']).sort_values(by=['word'])
        # group2 = meta_data_copy[meta_data_copy["isrelsent"] == False].sample(frac=1).drop_duplicates(subset=['word']).sort_values(by=['word'])
        #
        # stats.ttest_rel(group1["Value"], group2["Value"])

        return meta_data_copy[["Component", "Relevant term", "Value"]]

    # p300_eeg = extract_components(epos_list, meta_data, 0.2, 0.25, "P300")
    n400_eeg = extract_components(epos_list, meta_data, 0.3, 0.5, "N400")
    # p600_eeg = extract_components(epos_list, meta_data, 0.6, 0.7, "P600")
    # plot_data = pd.concat([p300_eeg, n400_eeg, p600_eeg])

    a4_dims = (8.27, 15)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.violinplot(
        ax=ax,
        data=n400_eeg,
        x="Component",
        y="Value",
        hue="Relevant term",
        split=True,
        orient="v",
        scale="count",
        inner="quartile",
    )
    if x_label is not None:
        ax.set_xlabel(xlabel=x_label)
    if y_label is not None:
        ax.set_ylabel(ylabel=y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    ax.legend(title="Relevant term")
    plt.title(title)
    plt.savefig(
        "{}/plots/results_{}.pdf".format(work_dir, file_id),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_correlation_ax(
    input_x,
    response_y,
    annotations,
    text_position,
    groups,
    with_text_annotations,
    corr_coef,
    p_value,
    is_legend_visible,
    ax,
    title,
    y_label,
    x_label,
    y_lim=None,
    x_lim=None,
    fontsize=24,
    s=800,
):

    response = copy.deepcopy(response_y)
    similarity = copy.deepcopy(input_x)

    if with_text_annotations and annotations is not None:
        group_colors = {1: "green", 0: "red"}
        ax = sns.regplot(ax=ax, x=similarity, y=response)
        # mask = list(map(bool, list(map(int, groups))))
        # ax = sns.regplot(ax=ax, x=similarity[mask], y=response[mask], scatter=False, fit_reg=True, line_kws=dict(color="g"))
        texts = []
        for x, y, s, label in zip(similarity, response, annotations, groups):
            texts.append(
                ax.text(
                    x, y, s, color=group_colors.get(label), fontsize=fontsize
                )
            )
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
        )
        if is_legend_visible:
            import matplotlib.patches as mpatches

            rel_patch = mpatches.Patch(
                color="green", label="Semantically salient words"
            )
            irrel_patch = mpatches.Patch(
                color="red", label="Semantically non-salient words"
            )
            ax.legend(handles=[rel_patch, irrel_patch], loc="upper right")
    else:
        if groups is not None:
            word_labels = []
            for x in groups:
                if x == 1:
                    word_label = "Semantically salient words"
                else:
                    word_label = "Semantically non-salient words"
                word_labels.append(word_label)
        else:
            word_labels = None

        ax = sns.regplot(ax=ax, x=similarity, y=response, scatter=False)
        ax = sns.scatterplot(
            ax=ax,
            x=similarity,
            s=s,
            y=response,
            hue=word_labels,
            alpha=0.5,
            palette={
                "Semantically salient words": "green",
                "Semantically non-salient words": "red",
            },
        )
        if is_legend_visible:
            handles, labels = ax.get_legend_handles_labels()
            if "non-" in labels[0]:
                order = [0, 1]
            else:
                order = [1, 0]
            ax.legend(
                [handles[idx] for idx in order],
                [labels[idx] for idx in order],
                loc="upper right",
                title="",
                ncol=1,
            )

        else:
            if ax.legend() is not None:
                ax.legend().remove()

    ax.set_xlabel(xlabel=x_label, labelpad=30)
    ax.set_ylabel(ylabel=y_label, labelpad=30)
    ax.set(xlim=x_lim, ylim=y_lim)
    ax.text(
        text_position[0],
        text_position[1],
        "Kendall’s Tau = {v:.2f}, \n p < {p:.2e}.".format(
            v=corr_coef, p=p_value
        ),
        fontstyle="italic",
        fontsize=fontsize,
    )
    ax.set_title(title)


def plot_correlation(
    ax,
    input_x,
    response_y,
    annotations,
    text_position,
    groups,
    with_text_annotations,
    is_legend_visible,
    title,
    y_label,
    x_label,
    x_lim=None,
    y_lim=None,
    fontsize=24,
    s=400,
):
    logging.info("The number of data points: {}".format(len(input_x)))

    # mask = list(map(bool, list(map(int, groups))))
    corr_coef, p_value = calculate_kendall_tau(input_x, response_y)

    plot_correlation_ax(
        input_x,
        response_y,
        annotations,
        text_position,
        groups,
        with_text_annotations,
        corr_coef,
        p_value,
        is_legend_visible=is_legend_visible,
        ax=ax,
        title=title,
        y_label=y_label,
        x_lim=x_lim,
        y_lim=y_lim,
        x_label=x_label,
        fontsize=fontsize,
        s=s,
    )


def plot_modality_scatterplot(
    x,
    y,
    annotations,
    file_id,
    work_dir,
    x_label=None,
    y_label=None,
    title=None,
    x_lim=None,
    y_lim=None,
    ax=None,
    y_label_right=False,
    fontsize=15,
):
    if ax is None:
        a4_dims = (15, 8.27)
        fig, ax = plt.subplots(figsize=a4_dims)
        fig.tight_layout()
    x_average = np.mean(x)
    y_average = np.mean(y)
    ax = sns.scatterplot(
        ax=ax,
        x=x,
        s=200,
        y=y,
        alpha=0.5,
    )

    texts = []
    scores_without_annotations = set()
    position = 0
    for x_, y_, s in zip(x, y, annotations):
        # Find the indices of the scores that are identical:
        identical_scores_mask = [
            x_ == xi and y_ == yi for xi, yi in zip(x, y)
        ]
        score_indices = np.array(
            [i for i in range(len(identical_scores_mask))]
        )
        identical_score_indices = score_indices[identical_scores_mask]

        # Annotations that overlap
        if position in scores_without_annotations:
            s = ""
        else:
            s = (
                [annotations[i] for i in identical_score_indices]
                .__str__()
                .replace("[", "")
                .replace("]", "")
            )
        texts.append(ax.text(x_, y_, s, fontsize=fontsize))
        scores_without_annotations.update(identical_score_indices)
        position += 1
    adjust_text(
        texts,
        ax=ax,
        x=x,
        y=y,
    )

    if x_label is not None:
        ax.set_xlabel(xlabel=x_label, labelpad=30)
    if y_label is not None:
        if y_label_right:
            ax2 = ax.twinx()
            ax2.set_ylabel(ylabel=y_label, labelpad=30)
        else:
            ax.set_ylabel(ylabel=y_label, labelpad=30)

    if x_lim is None:
        x_lim = (np.min(x) * 0.97, np.max(x) * 1.03)
    if y_lim is None:
        y_lim = (np.min(y) * 0.97, np.max(y) * 1.03)

    ax.axhline(y=y_average, color="r", linestyle="--")
    ax.axvline(x=x_average, color="r", linestyle="--")
    ax.set_yticks(np.arange(0.1, 1.01, 0.1))
    ax.set_yticklabels(np.arange(0.1, 1.01, 0.1))
    ax.set_xticks(np.arange(0.1, 1.01, 0.2))
    ax.set_xticklabels(np.arange(0.1, 1.01, 0.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title(title)

    if ax is None:
        plt.xticks(rotation=45)
        plt.title(title)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.title(title)
        plt.savefig(
            "{}/plots/results_{}.pdf".format(work_dir, file_id),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()
