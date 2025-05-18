import os.path
import torch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Optional
from doc_rel.model.embedding import get_embeddings


from doc_rel.configuration.configuration import Configuration
from doc_rel.misc.utils import (
    set_logging,
    set_seed,
    get_similarity_scores,
    create_folder,
)
from doc_rel.vis.model_performance import plot_correlation
from doc_rel.model.embedding import LANG_MODEL, parse_language_model
from doc_rel.data_operations.loader import DataProvider

OFFSETS = {
    "all": dict(x_offset=0.03, ytop=1.01),
    "atom": dict(x_offset=0.10, ytop=0.55),
}

# per-LM text_position tweaks
MODEL_CONFIGS = [
    {
        "id": "fasttext",
        "enum": LANG_MODEL.FAST_TEXT.value,
        "pos": {
            "all": [-0.06, 1.01 - 0.15],
            "atom": [0.15, 0.55 - 0.06],
        },
    },
    {
        "id": "bert",
        "enum": LANG_MODEL.BERT.value,
        "pos": {
            "all": [-0.06, 1.01 - 0.15],
            "atom": [0.00, 0.55 - 0.06],
        },
    },
    {
        "id": "sbert",
        "enum": LANG_MODEL.SBERT.value,
        "pos": {
            "all": [-0.06, 1.01 - 0.15],
            "atom": [-0.01, 0.55 - 0.06],
        },
    },
    {
        "id": "glove",
        "enum": LANG_MODEL.GLOVE.value,
        "pos": {
            "all": [-0.23, 1.01 - 0.15],
            "atom": [-0.01, 0.55 - 0.06],
        },
    },
    {
        "id": "word2vec",
        "enum": LANG_MODEL.WORD2VEC.value,
        "pos": {
            "all": [-0.06, 1.01 - 0.15],
            "atom": [-0.06, 0.55 - 0.06],
        },
    },
]


def prepare_data(
    conf: Configuration,
    predictions: pd.DataFrame,
    language_model: Optional[LANG_MODEL],
    document_relevance: int,
    round_values: bool,
    average_per_word: bool,
) -> pd.DataFrame:
    predictions = predictions[
        predictions["relevant_document"] == document_relevance
    ]
    predictions = predictions.reset_index(drop=True)
    predictions = predictions.groupby(
        by=["event", "topic", "word"], as_index=False
    ).mean()

    data_dir = os.path.join(conf.work_dir, "data_prepared_for_benchmark")
    provider = DataProvider(
        dir_data=data_dir,
        conf=conf,
        user=conf.users,
        language_model=language_model,
    )
    data = provider.data['text']
    data['emb_id'] = data.index
    data = data[["event", "topic", "word", "emb_id"]]

    df_merged = predictions.merge(
        data,
        how='left',
        on=["event", "topic", "word"],
    )

    topics_trimmed = (
        df_merged["topic"].apply(lambda x: x.replace(" ", "")).tolist()
    )

    topic_embeddings = torch.vstack(
        [
            get_embeddings(language_model, [topic], average=False).squeeze()
            for topic in topics_trimmed
        ]
    )

    word_embeddings = provider.data['embeddings'][df_merged['emb_id']]
    df_merged["similarity_scores"] = get_similarity_scores(
        word_embeddings, topic_embeddings
    )

    input_x = df_merged["similarity_scores"].tolist()
    response_y = df_merged["predicted_human_attention_soft"].tolist()
    annotations = df_merged["word"].tolist()
    labels = df_merged["semantic_relevance"].tolist()

    plot_df = pd.DataFrame(
        list(zip(input_x, response_y, labels, annotations)),
        columns=["x", "y", "labels", "annotations"],
    )
    if round_values:
        plot_df = plot_df.round({"x": 3})
        plot_df = plot_df.groupby(by=["x", "labels"], as_index=False).mean(
            numeric_only=True
        )

    if average_per_word:
        plot_df = plot_df.groupby("annotations", as_index=False).agg(
            x=("x", "mean"), y=("y", "mean"), labels=("labels", "mean")
        )

    return plot_df


def run(
    conf: Configuration,
    predictions_file: str,
    language_model: Optional[LANG_MODEL],
    topic: str,
    x_offset: float,
    ytop: float,
    text_position: list,
):
    prediction_data = pd.read_csv(predictions_file)

    if topic == "all":
        prediction_data = prediction_data
        show_annotations = False
        round_values = True
        average_per_word = False
    else:
        prediction_data = prediction_data[prediction_data["topic"] == topic]
        show_annotations = True
        round_values = False
        average_per_word = True
    columns = [
        "event",
        "word",
        "topic",
        "semantic_relevance",
        "predicted_human_attention_soft",
        "relevant_document",
    ]
    prediction_data = prediction_data[columns]

    data_relevant = prepare_data(
        conf=conf,
        predictions=prediction_data,
        language_model=language_model,
        document_relevance=1,
        round_values=round_values,
        average_per_word=average_per_word,
    )
    data_irrelevant = prepare_data(
        conf=conf,
        predictions=prediction_data,
        language_model=language_model,
        document_relevance=0,
        round_values=round_values,
        average_per_word=average_per_word,
    )

    a4_dims = (25, 12)
    fig, axes = plt.subplots(
        figsize=a4_dims, ncols=2, nrows=1, sharey=True, sharex=True
    )
    plt.subplots_adjust(wspace=0.03, hspace=0.03, bottom=0.15, left=0.08)
    title_relevant = (
        "Relevant documents" if topic == "all" else "Relevant document"
    )
    title_irrelevant = (
        "Irrelevant documents" if topic == "all" else "Irrelevant document"
    )
    x_min = np.min(
        [np.min(data_relevant['x']), np.min(data_irrelevant['x'])]
    )
    x_max = np.max(
        [np.max(data_relevant['x']), np.max(data_irrelevant['x'])]
    )
    for axis_id, data in enumerate([data_relevant, data_irrelevant]):
        plot_correlation(
            ax=axes[axis_id],
            input_x=data["x"].to_numpy(),
            response_y=data["y"].to_numpy(),
            annotations=(
                data["annotations"].tolist() if show_annotations else None
            ),
            groups=data["labels"].tolist(),
            is_legend_visible=True if axis_id == 1 else False,
            with_text_annotations=True if show_annotations else None,
            y_lim=[-0.03, ytop],
            x_lim=[x_min - 0.03, x_max + 0.03],
            text_position=text_position,
            title=title_relevant if axis_id == 0 else title_irrelevant,
            y_label="",
            x_label="",
            fontsize=20,
        )

    fig.supxlabel(
        "Cosine similarity between a document word and a topic name",
        fontsize=28,
    )
    fig.supylabel("Predicted human attention", fontsize=28)
    plt.savefig(
        f"{conf.work_dir}/plots/similarity_{topic}_{language_model}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    conf = Configuration()
    conf.selected_seed = 1
    set_seed(conf.selected_seed)
    set_logging(conf.work_dir, "correlation")
    predictions_folder = os.path.join(conf.work_dir, "predictions")
    create_folder(predictions_folder)

    for cfg in MODEL_CONFIGS:
        predictions_file = os.path.join(
            predictions_folder,
            f"eeg+text#language_model-{cfg['id']}#eeg_model-lda.csv",
        )
        for task in ("all", "atom"):
            run(
                conf,
                predictions_file,
                parse_language_model(cfg["enum"]),
                task,
                **OFFSETS[task],
                text_position=cfg["pos"][task],
            )
