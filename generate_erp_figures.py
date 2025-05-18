"""
To generate the ERP figures,
the raw EEG data must be first downloaded and cleaned.
The raw data can be downloaded from  https://osf.io/xh3g5/.

To acquire the cleaned data, follow the instructions
in the Section "Tutorial" (point 5) and "Data processing" (point 1 and 2)
in https://osf.io/xh3g5/wiki/home/.

Copy then or create a symbolic link to the cleaned data directory ("cleaned")
in the working directory.
"""

import os

from doc_rel.configuration.configuration import Configuration
from doc_rel.misc.utils import set_logging
from doc_rel.vis.eeg_data import loadepos, plot_erp


def run():
    # Get configuration:
    conf = Configuration()
    set_logging(conf.work_dir, "erp")

    cleaned_data_dir = os.path.join(conf.work_dir, "cleaned/")

    print("ERP: (a)")
    epos = loadepos(cleaned_data_dir)
    plot_erp(
        work_dir=conf.work_dir,
        epos=epos,
        title="Pz electrode.",
        queries=["semantic_relevance == 1", "semantic_relevance == 0"],
        file_id="word_relevance",
        l=["Semantically salient words", "Semantically non-salient words"],
    )

    print("ERP: (b)")
    epos = loadepos(cleaned_data_dir)
    plot_erp(
        work_dir=conf.work_dir,
        epos=epos,
        title="Pz electrode.",
        queries=["relevant_document==True", "relevant_document==False"],
        file_id="document_relevance",
        l=["Relevant documents", "Irrelevant documents"],
    )

    print("ERP: (c)")
    epos = loadepos(cleaned_data_dir)
    plot_erp(
        work_dir=conf.work_dir,
        epos=epos,
        title="Pz electrode.",
        queries=[
            "relevant_document==True and semantic_relevance==1",
            "relevant_document==False and semantic_relevance==1",
        ],
        file_id="document_relevance_semantic",
        l=["Relevant documents", "Irrelevant documents"],
    )


if __name__ == "__main__":
    run()
