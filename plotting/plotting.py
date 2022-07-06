import pandas as pd
import seaborn as sns
import pathlib
import h5py
import matplotlib.pyplot as plt
plt.rcParams['legend.title_fontsize'] = 'xx-small'
import numpy as np
from sklearn.metrics import roc_curve, auc
# import mplhep as hep
# plt.style.use(hep.style.CMS)

def get_roc_data(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val, drop_intermediate=False)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data

def get_mean_and_error(data):
    return [np.mean(data, axis=0), np.std(data, axis=0)]

def plot_ROC_kfold(quantum_loss_qcd, quantum_loss_sig, classic_loss_qcd, classic_loss_sig, ids, n_folds, colors, title, pic_id, xlabel='TPR', ylabel='1/FPR', legend_loc='best', legend_title='$ROC$', save_dir=None):

    fig = plt.figure(figsize=(10,8))

    for i in range(len(ids)): # for each latent space or train size
        fpr_q=[]; fpr_c=[]
        auc_q=[]; auc_c=[]
        tpr_q=[]; tpr_c=[]
        for j in range(n_folds):
            # quantum data
            fq, tq, _ = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j])
            # classic data
            fc, tc, _ = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j])

            auc_q.append(auc(fq, tq)); auc_c.append(auc(fc, tc))
            #auc_q.append(get_auc(fq, tq)); auc_c.append(get_auc(fc, tc))
            fpr_q.append(1./np.array(fq)); fpr_c.append(1./np.array(fc))
            tpr_q.append(tq); tpr_c.append(tc)
        
        auc_data_q = get_mean_and_error(np.array(auc_q))
        auc_data_c = get_mean_and_error(np.array(auc_c))
        
        fpr_data_q = get_mean_and_error(np.array(fpr_q))
        fpr_data_c = get_mean_and_error(np.array(fpr_c))
        
        tpr_mean_q = np.mean(np.array(tpr_q), axis=0)
        tpr_mean_c = np.mean(np.array(tpr_c), axis=0)

        #plt.fill_between(x, y-error, y+error)
        plt.fill_between(tpr_mean_q, fpr_data_q[0]-fpr_data_q[1], fpr_data_q[0]+fpr_data_q[1], alpha=0.8, label='(%s) Quantum: (auc = %.2f+/-%.2f)'% (ids[i], auc_data_q[0]*100., auc_data_q[1]*100.))
        plt.fill_between(tpr_mean_c, fpr_data_c[0]-fpr_data_c[1], fpr_data_c[0]+fpr_data_c[1], alpha=0.5, label='(%s) Classic: (auc = %.2f+/-%.2f)'% (ids[i], auc_data_c[0]*100., auc_data_c[1]*100.))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.yscale('log')
    plt.xlim(0.0, 1.0)
    plt.title(title)
    leg = plt.legend(fancybox=True, frameon=True, prop={"size":10}, bbox_to_anchor =(1.0, 1.0))
    leg.get_title().set_position((-40, 0))
    fig.tight_layout()
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/ROC_final_{pic_id}.pdf', dpi = fig.dpi, bbox_inches='tight')
    else: plt.show()