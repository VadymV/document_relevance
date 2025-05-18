# Predicting document relevance from brain recordings

## Configure the environment

``uv`` is used for dependency management. See how to install ``uv`` [here][2].
Run ``uv sync`` in the folder where the ``README.md`` file is located.

## Data preparation
There are 2 options:
1. Download the ``data_prepared_for_benchmark`` from [here][3] and extract the files.
This option is the fastest, as the data are already preprocessed and prepared for benchmarking.
2. A new virtual environment must be used to avoid any conflicts with the
existing packages. Download the *raw* data and annotations.csv from [here][5].
Follow the instructions in the Section "Tutorial" (point 5).
Run the following commands:

```py
    from doc_rel.releegance.data_operations.preprocessor import DataPreprocessor
    from doc_rel.releegance.data_operations.preparator import DataPreparator
    
    data_preprocessor = DataPreprocessor(project_path=project_path)
    data_preprocessor.filter()
    data_preprocessor.create_epochs()
    data_preprocessor.clean()
    
    data_preparator = DataPreparator(
        data_dir=data_preprocessor.cleaned_data_dir)
    data_preparator.prepare_data_for_benchmark()
```
where ``project_path`` points to the folder that contains the *raw* data and annotations.csv.
After running this script, the folder ``data_prepared_for_benchmark``
with the prepared data for benchmarking will be created in the project_path.


## Run experiments

- All settings are defined in ``configuration.yaml``. The folder ``data_prepared_for_benchmark``
must be copied to ``work_dir`` in ``configuration.yaml`` or a symbolic link must be created.
- Run the experiments:``uv run python run_experiments.py --eeg_model=EM --language_model=LM --experiment=EXPERIMENT``, where EM, LM, EXPERIMENT is the desired parameters. See help for more details.

## Generate figures
- ``uv run python generate_latex_tables.py`` will generate Tables 1-4 in the paper.
- ``uv run python generate_correleation_figures.py`` will generate figures 7 and 8 in the paper (+ Supplementary).
- ``uv run python generate_erp_figures.py`` will generate Figure 3 in the paper.
- ``uv run python generate_relevance_figures.py`` will generate Figures 5 and 6 in the paper (+ Supplementary).

---

For citations, please use:
```
@unpublished{Gryshchuk2025_document-relevance,
   author = {Vadym Gryshchuk and Maria Maistro and Christina Lioma and Tuukka Ruotsalo},
   title = {Predicting Document Relevance from Brain Recordings},
   year = {2025},
   note = {under review (TOIS)}
}
```

  [1]: https://huggingface.co/datasets/Quoron/EEG-semantic-text-relevance
  [2]: https://docs.astral.sh/uv/getting-started/installation/
  [3]: https://drive.proton.me/urls/2TWQXJW2C4#9G2lbi7SuGFE
  [4]: https://arxiv.org/abs/1910.10781
  [5]: https://osf.io/xh3g5/files/osfstorage