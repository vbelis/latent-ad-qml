import math
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import h5py


def euclidean_dist(a, b):
    return np.linalg.norm(a - b)


def calc_norm(a, b):
    return math.sqrt(np.sum(a**2) + np.sum(b**2))


def combine_loss_min(loss):
    """Returns minimum loss for 2 jet data."""
    loss_j1, loss_j2 = np.split(loss, 2)
    return np.minimum(loss_j1, loss_j2)


def load_clustering_test_data(
    lat_dim,
    test_size=10000,
    k=2,
    signal_name="RSGraviton_WW",
    mass="35",
    br_na=None,
    around_peak=None,
    read_dir="/eos/user/e/epuljak/private/epuljak/public/diJet",
):
    """Load data for evaluation/testing.

    Parameters
    ----------
    lat_dim : int
        Latent dimension of input data.
    test_size : int
        Number of samples for testing.
    k : int
        Number of classes in quantum/classical k-medians.
    signal_name : str
        Name of the new physics process used for testing.
    mass : str
        String number of mass connected to specified signal.
    br_na : str, or `None`
        Specifiying "Broad" as "BR", or "Narrow" as "NA" signal.
    around_peak : str
        String specifying around which mass peak we are performing testing analysis.
    read_dir : str
        Name of directory from where to load all testing data.

    Returns
    -------
    numpy.ndarray
        Test QCD data
    numpy.ndarray
        Test BSM data
    """
    # read QCD data
    file_name = f"{read_dir}/{lat_dim}/latentrep_QCD_sig_test.h5"
    with h5py.File(file_name, "r") as file:
        data = file["latent_space"]
        l1 = data[:, 0, :]
        l2 = data[:, 1, :]
        data_test_qcd = np.vstack([l1[:test_size], l2[:test_size]])

    # read SIGNAL data
    read_dir = f"{read_dir}/{lat_dim}"
    if br_na:
        signal = f"{signal_name}_{br_na}_{mass}"
    else:
        signal = f"{signal_name}_{mass}"
    if around_peak:
        file_name = f"{read_dir}/latentrep_{signal}_{around_peak}.h5"
    else:
        file_name = f"{read_dir}/latentrep_{signal}.h5"
    with h5py.File(file_name, "r") as file:
        data = file["latent_space"]
        l1 = data[:, 0, :]
        l2 = data[:, 1, :]
        data_test_sig = np.vstack([l1[:test_size], l2[:test_size]])

    return data_test_qcd, data_test_sig


def ad_score(cluster_assignments, distances, method="sum_all"):
    """Calculated anomaly detection score = distance to cluster medians.

    Parameters
    ----------
    cluster_assignments: numpy.ndarray, or list
        Cluster assignments for all data points.
    distances : numpy.ndarray, or list
        Distances for each data point to assigned cluster median.
    method : str
        'sum_all' = sum all distances of each data point to assigned cluster median, or
        sum distances of points assigned to specific cluster

    Returns
    -------
    numpy.ndarray(dtype=float)
        Anomaly detection score
    """
    if method == "sum_all":
        return np.sqrt(np.sum(distances**2, axis=1))
    else:
        return np.sqrt(distances[range(len(distances)), cluster_assignments] ** 2)


def get_auc(qcd, bsm):
    """Calculates AUC

    Parameters
    ----------
    qcd : numpy.ndarray, or list
        QCD, or normal data
    bsm : numpy.ndarray, or list
        BSM, or anomalous data

    Returns
    -------
    float
        AUC
    """
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)

    return auc_data


def get_metric(qcd, bsm, tpr_window=[0.5, 0.6]):
    """Calculates value of 1/FPR and error.

    Parameters
    ----------
    qcd : numpy.ndarray, or list
        QCD, or normal data
    bsm : numpy.ndarray, or list
        BSM, or anomalous data
    tpr_window : [float, float]
        Window of fixed TPR for which to calculate FPR.

    Returns
    -------
    numpy.ndarray, or list (dtype=float)
        1/fpr data
    numpy.ndarray, or list (dtype=float)
        1/fpr error
    """

    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)

    # FPR and its error
    position = np.where((tpr_loss >= tpr_window[0]) & (tpr_loss <= tpr_window[1]))[0][0]
    threshold_data = threshold_loss[position]
    pred_data = [1 if i >= threshold_data else 0 for i in list(pred_val)]

    tn, fp, fn, tp = confusion_matrix(true_val, pred_data).ravel()

    fpr_data = fp / (fp + tn)
    one_over_fpr_data = 1.0 / fpr_data  # y = 1/x

    fpr_error = np.sqrt(fpr_data * (1 - fpr_data) / (fp + tn))
    one_over_fpr_error = fpr_error * (
        1.0 / np.power(fpr_data, 2)
    )  # sigma_y = sigma_x * (1/x^2)

    return one_over_fpr_data, one_over_fpr_error


def calc_AD_scores(
    identifiers,
    n_samples_train,
    test_size=10000,
    signal_name="RSGraviton_WW",
    mass="35",
    br_na=None,
    around_peak=None,
    q_dir=None,
    c_dir=None,
    read_test_dir=None,
    save_dir=None,
    classic=True,
):
    """Calculates anomaly detection scores for different study - identified by identifiers.

    Parameters
    ----------
    identifiers : list(str)
        List of identifiers explaining the study.
    n_samples_train : list(str)
        List of string number indicating number of training samples for each study.
    test_size : int
        Number of test samples.
    signal_name : str
        Name of the new physics process used for testing.
    mass : str
        String number of mass connected to specified signal.
    br_na : str, or `None`
        Specifiying "Broad" as "BR", or "Narrow" as "NA" signal.
    around_peak : str
        String specifying around which mass peak we are performing testing analysis.
    q_dir : str
        Name of directory from where to load centroids found by quantum k-medians.
    c_dir : str
        Name of directory from where to load centroids found by classical k-medians.
    read_test_dir : str
        Name of directory from where to load test data.
    save_dir : str
        Name of directory where to save anomaly detection scores.

    Returns
    -------
    list
        Anomaly detection scores calculated for quantum k-medians.
    list
        Anomaly detection scores calculated for classical k-medians.
    """
    quantum = []
    classic = []

    for i in range(len(identifiers)):  # for each latent space or train size

        # load q-centroids
        centroids_q = np.load(
            f"{q_dir}/centroids_lat{identifiers[i]}_{n_samples_train[i]}.npy"
        )
        # load c-centroids
        centroids_c = np.load(
            f"{c_dir}/centroids_lat{identifiers[i]}_{n_samples_train[i]}.npy"
        )
        test_qcd, test_sig = u.load_clustering_test_data(
            identifiers[i],
            test_size=test_size,
            k=2,
            signal_name=signal_name,
            mass=mass,
            br_na=br_na,
            read_dir=read_test_dir,
            around_peak=around_peak,
        )

        # find cluster assignments + distance to centroids for test data
        q_cluster_assign, q_distances = qkmed.find_nearest_neighbour_DI(
            test_qcd, centroids_q
        )
        q_cluster_assign_s, q_distances_s = qkmed.find_nearest_neighbour_DI(
            test_sig, centroids_q
        )
        c_cluster_assign, c_distances = cf.find_nearest_neighbour_classic(
            test_qcd, centroids_c
        )
        c_cluster_assign_s, c_distances_s = cf.find_nearest_neighbour_classic(
            test_sig, centroids_c
        )

        # calc AD scores
        q_score_qcd = u.ad_score(q_cluster_assign, q_distances)
        q_score_sig = u.ad_score(q_cluster_assign_s, q_distances_s)
        c_score_qcd = u.ad_score(c_cluster_assign, c_distances)
        c_score_sig = u.ad_score(c_cluster_assign_s, c_distances_s)

        # calculate loss from 2 jets
        quantum_loss_qcd = u.combine_loss_min(q_score_qcd)
        quantum_loss_sig = u.combine_loss_min(q_score_sig)
        quantum.append([quantum_loss_qcd, quantum_loss_sig])

        classic_loss_qcd = u.combine_loss_min(c_score_qcd)
        classic_loss_sig = u.combine_loss_min(c_score_sig)
        classic.append([classic_loss_qcd, classic_loss_sig])

        # save losses
        data_save = pd.DataFrame(
            {
                "quantum_loss_qcd": quantum_loss_qcd,
                "quantum_loss_sig": quantum_loss_sig,
                "classic_loss_qcd": classic_loss_qcd,
                "classic_loss_sig": classic_loss_sig,
            }
        )

        data_save.to_pickle(
            f"{save_dir}/Latent_{identifiers[i]}_trainsize_{n_samples_train[i]}_{signal_name}{mass}{br_na}.pkl"
        )

    return quantum, classic


def get_roc_data(qcd, bsm):
    """Calculates FPR, TPR and AUC"""
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data
