# Create plots ROC plots for the paper.


from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import roc_curve, auc
import mplhep as hep
from matplotlib.lines import Line2D


def get_roc_data(
    qcd: np.ndarray, bsm: np.ndarray, fix_tpr: bool = False
) -> Tuple[np.ndarray]:
    """Compute roc curves given the background and anomaly datasets.

    Parameters
    ----------
    qcd : np.ndarray
        Background QCD dataset.
    bsm : np.ndarray
        Anomaly, Beyond the Standard Model (BSM) dataset.
    fix_tpr : bool, optional
        Constant threshold selection for ROC curve calculation, by default False

    Returns
    -------
    Tuple
        np.ndarray
        False Positive Rate array.
        np.ndarray
        True Positive Rate array.
    """
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(
        true_val, pred_val, drop_intermediate=False
    )
    if fix_tpr:
        return fpr_loss, tpr_loss, threshold_loss, true_val, pred_val
    return fpr_loss, tpr_loss


def get_FPR_for_fixed_TPR(
    tpr_window: float, fpr_loss: np.ndarray, tpr_loss: np.ndarray, tolerance: float
) -> float:
    """Get FPR for a fixed value of TPR.

    Calculation of the ROC curve is in discrete steps. A window of tolerance is defined
    around the desired TPR working point and the mean of FPR is taken there.

    Parameters
    ----------
    tpr_window : float
        TPR working point, typically 0.6 or 0.8
    fpr_loss : np.ndarray
        FPR array of the ROC curve.
    tpr_loss : np.ndarray
        TPR array of the ROC curve.
    tolerance : float
        Tolerance around working point. 0.1-1% window.

    Returns
    -------
    float
        Mean FPR at the tolerance window around the TPR working point
    """
    position = np.where(
        (tpr_loss >= tpr_window - tpr_window * tolerance)
        & (tpr_loss <= tpr_window + tpr_window * tolerance)
    )[0]

    return np.mean(fpr_loss[position])


def get_mean_and_error(data: np.ndarray) -> Tuple[float]:
    """Compute the mean and std of an array.

    Parameters
    ----------
    data : np.ndarray
        The input array.

    Returns
    -------
    Tuple
        float:
            The mean.
        float:
            The standard deviation.
    """
    return [np.mean(data, axis=0), np.std(data, axis=0)]


def plot_ROC_kfold_mean(
    quantum_loss_qcd: List[np.ndarray],
    quantum_loss_sig: List[np.ndarray],
    classic_loss_qcd: List[np.ndarray],
    classic_loss_sig: List[np.ndarray],
    ids: List[str],
    n_folds: int,
    pic_id: str = None,
    xlabel: str = "TPR",
    ylabel: str = r"1/FPR",
    legend_title: str = "$ROC$",
    save_dir: str = None,
    palette: List[str] = ["#3E96A1", "#EC4E20", "#FF9505"],
):
    """Calculate the mean ROC curve and its std uncertainty band.

    Using the scores of the the classical and quantum models, the ROC curves are
    computed for each on of the k-folds. The mean and std is computed, and the ROC
    mean ROC curve is plotted with its error band. The AUC mean and std is also
    calculated and presented in the legend of the figure.

    Parameters
    ----------
    quantum_loss_qcd : List[np.ndarray]
        List of scores of the quantum model on the background (QCD) data.
    quantum_loss_sig : List[np.ndarray]
        List of scores of the quantum model on the signal (anomaly) data.
    classic_loss_qcd : List[np.ndarray]
        List of scores of the classical model on the background (QCD) data.
    classic_loss_sig : List[np.ndarray]
        List of scores of the classical model on the signal (anomaly) data.
    ids : List[str]
        Identifier of the different scores corresponing to the lists of scores.
        Namely, 3 different anomalies, 3 different latent dimensions or
        3 different training sizes.
    n_folds : int
        Number of k-folds.
    pic_id : str, optional
        Name of the output figure, by default None
    xlabel : str, optional
        Label for the x-axis of the figure, by default "TPR"
    ylabel : str, optional
        Label for the y-axis of the figure, by default r"1/FPR"
    legend_title : str, optional
        Title of the main legend, by default "$"
    save_dir : str, optional
        Output directory for the produced figure, by default None
    palette : List[str], optional
        Colors for the 3 ROC curves per plot based on the ids,
        by default ["#3E96A1", "#EC4E20", "#FF9505"]
    """

    styles = ["solid", "dashed"]
    plt.style.use(hep.style.CMS)
    fig = plt.figure(figsize=(8, 8))
    anomaly_auc_legend = []
    study_legend = []
    for i, id_name in enumerate(ids):  # for each latent space or train size
        fpr_q = []
        fpr_c = []
        auc_q = []
        auc_c = []
        tpr_q = []
        tpr_c = []
        for j in range(n_folds):
            # quantum data
            fq, tq = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j])
            # classic data
            fc, tc = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j])

            auc_q.append(auc(fq, tq))
            auc_c.append(auc(fc, tc))
            fpr_q.append(fq)
            fpr_c.append(fc)
            tpr_q.append(tq)
            tpr_c.append(tc)

        auc_data_q = get_mean_and_error(np.array(auc_q))
        auc_data_c = get_mean_and_error(np.array(auc_c))

        fpr_data_q = get_mean_and_error(1.0 / np.array(fpr_q))
        fpr_data_c = get_mean_and_error(1.0 / np.array(fpr_c))

        tpr_mean_q = np.mean(np.array(tpr_q), axis=0)
        tpr_mean_c = np.mean(np.array(tpr_c), axis=0)

        if (
            ids[i] == "Narrow " r"G $\to$ WW 3.5 TeV"
        ):  # uncertainties are bigger for G_NA
            band_ind = np.where(tpr_mean_q > 0.6)[0]
        else:
            band_ind = np.where(tpr_mean_q > 0.35)[0]

        plt.plot(tpr_mean_q, fpr_data_q[0], linewidth=1.5, color=palette[i])
        plt.plot(
            tpr_mean_c,
            fpr_data_c[0],
            linewidth=1.5,
            color=palette[i],
            linestyle="dashed",
        )
        plt.fill_between(
            tpr_mean_q[band_ind],
            fpr_data_q[0][band_ind] - fpr_data_q[1][band_ind],
            fpr_data_q[0][band_ind] + fpr_data_q[1][band_ind],
            alpha=0.2,
            color=palette[i],
        )
        plt.fill_between(
            tpr_mean_c[band_ind],
            fpr_data_c[0][band_ind] - fpr_data_c[1][band_ind],
            fpr_data_c[0][band_ind] + fpr_data_c[1][band_ind],
            alpha=0.2,
            color=palette[i],
        )
        anomaly_auc_legend.append(
            f" {auc_data_q[0]*100:.2f}"
            f"± {auc_data_q[1]*100:.2f} "
            f"| {auc_data_c[0]*100:.2f}"
            f"± {auc_data_c[1]*100:.2f}"
        )
        study_legend.append(id_name)
    dummy_res_lines = [
        Line2D([0, 1], [0, 1], linestyle=s, color="black") for s in styles[:2]
    ]
    lines = plt.gca().get_lines()
    plt.semilogy(
        np.linspace(0, 1, num=int(1e4)),
        1.0 / np.linspace(0, 1, num=int(1e4)),
        linewidth=1.5,
        linestyle="--",
        color="0.75",
    )
    legend1 = plt.legend(
        dummy_res_lines,
        [r"Quantum", r"Classical"],
        frameon=False,
        loc="upper right",
        handlelength=1.5,
        fontsize=16,
        title_fontsize=14,
    )  # , bbox_to_anchor=(0.01,0.65)) # bbox_to_anchor=(0.97,0.78) -> except for latent study
    legend2 = plt.legend(
        [lines[i * 2] for i in range(len(palette))],
        anomaly_auc_legend,
        loc="lower left",
        frameon=True,
        title=r"AUC $\;\quad$Quantum $\quad\;\;\;$ Classical",
        fontsize=15,
        title_fontsize=14,
        markerscale=0.5,
    )
    legend3 = plt.legend(
        [lines[i * 2] for i in range(len(palette))],
        study_legend,
        markerscale=0.5,
        loc="center right",
        frameon=True,
        title=legend_title,
        fontsize=14,
        title_fontsize=15,
        bbox_to_anchor=(0.95, 0.75),
    )
    legend3.get_frame().set_alpha(0.35)

    legend1._legend_box.align = "left"
    legend2._legend_box.align = "left"
    legend3._legend_box.align = "center"

    for leg in legend1.legendHandles:
        leg.set_linewidth(2.2)
        leg.set_color("gray")
    for leg in legend2.legendHandles:
        leg.set_linewidth(2.2)

    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.gca().add_artist(legend3)
    plt.ylabel(ylabel, fontsize=24)
    plt.xlabel(xlabel, fontsize=24)
    plt.yscale("log")
    plt.xlim(0.0, 1.05)
    fig.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/ROC_{pic_id}.pdf", dpi=fig.dpi, bbox_inches="tight")
    else:
        plt.show()


def create_table_for_fixed_TPR(
    quantum_loss_qcd: List[np.ndarray],
    quantum_loss_sig: List[np.ndarray],
    classic_loss_qcd: List[np.ndarray],
    classic_loss_sig: List[np.ndarray],
    ids: List[str],
    n_folds: int,
    tpr_windows: List[float] = [0.4, 0.6, 0.8],
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Compute mean and std of FPR @FPR working point.

    Parameters
    ----------
    quantum_loss_qcd : List[np.ndarray]
        List of scores of the quantum model on the background (QCD) data.
    quantum_loss_sig : List[np.ndarray]
        List of scores of the quantum model on the signal (anomaly) data.
    classic_loss_qcd : List[np.ndarray]
        List of scores of the classical model on the background (QCD) data.
    classic_loss_sig : List[np.ndarray]
        List of scores of the classical model on the signal (anomaly) data.
    ids : List[str]
        Identifier of the different scores corresponing to the lists of scores.
        Namely, 3 different anomalies, 3 different latent dimensions or
        3 different training sizes.
    n_folds : int
        Number of k-folds.
    tpr_windows : List[float]
        TPR working point, by default [0.4, 0.6, 0.8]
    tolerance : float
        Tolerance around working point, by default 0.01

    Returns
    -------
    pd.DataFrame
        Latex table of the results.
    """
    # output to latex
    df_output = pd.DataFrame({"1/FPR": ["Quantum", "Classic"]})
    df_delta = pd.DataFrame(columns=["q_model", "tpr", "delta_qc", "delta_qc_unc"])
    for i in range(len(ids)):  # for each latent space or train size
        fpr_q = {f"{tpr}": [] for tpr in tpr_windows}
        fpr_c = {f"{tpr}": [] for tpr in tpr_windows}
        for j in range(n_folds):
            # quantum data
            fq, tq, _, true_q, pred_q = get_roc_data(
                quantum_loss_qcd[i][j], quantum_loss_sig[i][j], fix_tpr=True
            )
            # classic data
            fc, tc, _, true_c, pred_c = get_roc_data(
                classic_loss_qcd[i][j], classic_loss_sig[i][j], fix_tpr=True
            )
            for window in tpr_windows:
                f_q = get_FPR_for_fixed_TPR(
                    window, np.array(fq), np.array(tq), true_q, pred_q, tolerance
                )
                f_c = get_FPR_for_fixed_TPR(
                    window, np.array(fc), np.array(tc), true_c, pred_c, tolerance
                )
                fpr_q[f"{window}"].append(f_q)
                fpr_c[f"{window}"].append(f_c)

        for window in tpr_windows:

            fpr_data_q = get_mean_and_error(1.0 / np.array(fpr_q[f"{window}"]))
            fpr_data_c = get_mean_and_error(1.0 / np.array(fpr_c[f"{window}"]))

            fpr_over_data_q = fpr_data_q[0]
            fpr_over_data_c = fpr_data_c[0]

            fpr_over_error_q = fpr_data_q[1]
            fpr_over_error_c = fpr_data_c[1]

            df_output[f"TPR={window}"] = [
                f"{fpr_over_data_q:.2f} +/- {fpr_over_error_q:.2f}",
                f"{fpr_over_data_c:.2f} +/- {fpr_over_error_c:.2f}",
            ]
            if window != 0.4:
                # delta_qc = (fpr_over_data_q-fpr_over_data_c)/fpr_over_data_c
                delta_qc = fpr_over_data_q / fpr_over_data_c
                delta_qc_unc = math.sqrt(
                    (fpr_over_error_q / fpr_over_data_c) ** 2
                    + (fpr_over_data_q / fpr_over_data_c**2 * fpr_over_error_c) ** 2
                )
                d = {
                    "q_model": ids[i],
                    "tpr": window,
                    "delta_qc": delta_qc,
                    "delta_qc_unc": delta_qc_unc,
                }
                df_delta = df_delta.append(d, ignore_index=True)
        print(
            df_output.to_latex(
                index=False,
                caption=f"Latent space: {ids[i]}, TPR value +/- {tolerance*100}\%",
            )
        )

    return df_delta
