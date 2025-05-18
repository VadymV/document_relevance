import copy
import random
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import mne
import numpy as np


def loadepos(indir, decimate=None):
    epos = []
    for i, f in enumerate([f for f in sorted(listdir(indir)) if ".fif" in f]):
        epo = mne.read_epochs(join(indir, f))
        if decimate:
            epo = epo.decimate(decimate, verbose=True)
        epo.interpolate_bads(verbose=False)
        epo.pick_types(eeg=True)
        a = epo.metadata
        a["subject"] = "S%02d" % (i + 1)
        a["Subject"] = f.split("-")[0]
        a["relevant_document"] = a["selected_topic"] == a["topic"]
        epo.metadata = a
        epos.append(epo)

    return epos


def _perm_fun(x: np.ndarray, n_test: int, n_control: int):
    """
    The direction is 'test minus control'.
    Args:
        x:
        n_test:
        n_control:

    Returns:

    """
    n = n_test + n_control
    idx_control = set(random.sample(range(n), n_control))
    idx_test = set(range(n)) - idx_control
    return x[list(idx_test)].mean() - x[list(idx_control)].mean()


def run_permutation_test(control, test):
    a = copy.deepcopy(control)
    b = copy.deepcopy(test)
    observed_difference = np.mean(b) - np.mean(a)
    perm_diffs = [
        _perm_fun(np.concatenate((a, b), axis=None), a.shape[0], b.shape[0])
        for _ in range(10000)
    ]
    p = np.mean([diff > observed_difference for diff in perm_diffs])
    print("P value: {:.4f}".format(p))

    return p


# ERP PLOTS


def plot_erp(
    work_dir,
    epos,
    queries,
    file_id,
    ch_names=["Pz"],
    l=None,
    is_windowed=False,
    title=None,
    **kwargs,
):
    epos_copy = copy.deepcopy(epos)
    l = l or queries
    evos = {l[0]: [], l[1]: []}

    if "window_size" in kwargs:
        trials = 50
    else:
        trials = 1

    # predictions = read_predictions_files("D:/Data/Eugster/work_dir/predictions/#modality-EEG+Text_language_model-1#/")
    for i in range(trials):
        for idx, epo in enumerate(epos_copy, 1):
            epoch_copy = epo.copy()
            # epoch_copy.metadata = fill_with_language_predictions(epoch_copy.metadata, predictions)
            chi = [
                epoch_copy.info["ch_names"].index(ch_name)
                for ch_name in ch_names
            ]
            if (
                np.isnan(
                    epoch_copy[queries[0]]
                    .average(picks=chi)
                    .crop(tmin=0, tmax=1)
                    .data
                ).any()
                or np.isnan(
                    epoch_copy[queries[1]]
                    .average(picks=chi)
                    .crop(tmin=0, tmax=1)
                    .data
                ).any()
            ):
                continue
            evos[l[0]].append(
                epoch_copy[queries[0]].average(picks=chi).crop(tmin=0, tmax=1)
            )
            evos[l[1]].append(
                epoch_copy[queries[1]].average(picks=chi).crop(tmin=0, tmax=1)
            )

    run_permutation_test(
        test=np.mean(
            [x.get_data().squeeze() for x in list(evos.items())[0][1]], axis=0
        ),
        control=np.mean(
            [x.get_data().squeeze() for x in list(evos.items())[1][1]], axis=0
        ),
    )

    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
    font = {"size": 14}
    mpl.rc("font", **font)
    _, axs = plt.subplots(
        nrows=1,
        ncols=1,
    )
    fig = mne.viz.plot_compare_evokeds(
        evos,
        picks=ch_names,
        title=title,
        vlines=[0.25, 0.95],
        colors={
            l[1]: "xkcd:red",
            l[0]: "xkcd:green",
        },
        linestyles=["solid", "dotted"],
        ylim=dict(eeg=[-4, 4]),
        show=False,
        legend="upper center",
        show_sensors="lower left",
        axes=axs,
    )
    for item in axs.get_xticklabels():
        item.set_rotation(45)
    plt.savefig(
        "{}/plots/erp_{}.pdf".format(work_dir, file_id),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
    mpl.rcParams.update(mpl.rcParamsDefault)
