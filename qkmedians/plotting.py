import pandas as pd
import h5py
import matplotlib.pyplot as plt
plt.rcParams['legend.title_fontsize'] = 'xx-small'
import numpy as np
from sklearn.metrics import roc_curve, auc
# import mplhep as hep
# plt.style.use(hep.style.CMS)   

def get_roc_data(qcd, bsm, fix_tpr=False):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val, drop_intermediate=False)
    if fix_tpr: return fpr_loss, tpr_loss, threshold_loss, true_val, pred_val
    return fpr_loss, tpr_loss

def get_FPR_for_fixed_TPR(tpr_window, fpr_loss, tpr_loss, true_data, pred_data, tolerance):
    position = np.where((tpr_loss>=tpr_window-tpr_window*tolerance) & (tpr_loss<=tpr_window+tpr_window*tolerance))[0]
    return np.mean(fpr_loss[position])

def get_mean_and_error(data):
    return [np.mean(data, axis=0), np.std(data, axis=0)]

def plot_ROC_kfold_mean(quantum_loss_qcd, quantum_loss_sig, classic_loss_qcd, classic_loss_sig, ids, n_folds, colors, title, pic_id=None, xlabel='TPR', ylabel='1/FPR', legend_loc='best', legend_title='$ROC$', save_dir=None):

    fig = plt.figure(figsize=(10,8))

    for i in range(len(ids)): # for each latent space or train size
        fpr_q=[]; fpr_c=[]
        auc_q=[]; auc_c=[]
        tpr_q=[]; tpr_c=[]
        for j in range(n_folds):
            # quantum data
            fq, tq = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j])
            # classic data
            fc, tc = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j])

            auc_q.append(auc(fq, tq)); auc_c.append(auc(fc, tc))
            fpr_q.append(fq); fpr_c.append(fc)
            tpr_q.append(tq); tpr_c.append(tc)
        
        auc_data_q = get_mean_and_error(np.array(auc_q))
        auc_data_c = get_mean_and_error(np.array(auc_c))
        
        fpr_data_q = get_mean_and_error(np.array(fpr_q))
        fpr_data_c = get_mean_and_error(np.array(fpr_c))
        
        tpr_mean_q = np.mean(np.array(tpr_q), axis=0)
        tpr_mean_c = np.mean(np.array(tpr_c), axis=0)
        
        fpr_over_data_q = 1./fpr_data_q[0]
        fpr_over_data_c = 1./fpr_data_c[0]

        fpr_over_error_q = fpr_data_q[1]*(1./np.power(fpr_data_q[0],2)) # sigma_x*(1/x^2)
        fpr_over_error_c = fpr_data_c[1]*(1./np.power(fpr_data_c[0],2))
        
        plt.plot(tpr_mean_q, fpr_over_data_q, label='(%s) Quantum: (auc = %.2f+/-%.2f)'% (ids[i], auc_data_q[0]*100., auc_data_q[1]*100.), linewidth=1.5, color=colors[i])
        #plt.fill_between(tpr_mean_q, fpr_over_data_q-fpr_over_error_q, fpr_over_data_q+fpr_over_error_q, alpha=0.8, label='(%s) Quantum: (auc = %.2f+/-%.2f)'% (ids[i], auc_data_q[0]*100., auc_data_q[1]*100.))
        #plt.fill_between(tpr_mean_c, fpr_data_c[0]-fpr_data_c[1], fpr_data_c[0]+fpr_data_c[1], alpha=0.5, label='(%s) Classic: (auc = %.2f+/-%.2f)'% (ids[i], auc_data_c[0]*100., auc_data_c[1]*100.))
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

def create_table_for_fixed_TPR(quantum_loss_qcd, quantum_loss_sig, classic_loss_qcd, classic_loss_sig, ids, n_folds, tpr_windows=[0.4, 0.6, 0.8], tolerance=0.01):
    
    # output to latex
    df_output = pd.DataFrame({'1/FPR': ['Quantum', 'Classic']})
    
    for i in range(len(ids)): # for each latent space or train size
        fpr_q={f'{tpr}':[] for tpr in tpr_windows}; fpr_c={f'{tpr}':[] for tpr in tpr_windows}
        for j in range(n_folds):
            # quantum data
            fq, tq, _, true_q, pred_q = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j], fix_tpr=True)
            # classic data
            fc, tc, _, true_c, pred_c = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j], fix_tpr=True)
            
            for window in tpr_windows:
                f_q = get_FPR_for_fixed_TPR(window, np.array(fq), np.array(tq), true_q, pred_q, tolerance)
                f_c = get_FPR_for_fixed_TPR(window, np.array(fc), np.array(tc), true_c, pred_c, tolerance)
                fpr_q[f'{window}'].append(f_q)
                fpr_c[f'{window}'].append(f_c)
        
        for window in tpr_windows:
            
            fpr_data_q = get_mean_and_error(np.array(fpr_q[f'{window}']))
            fpr_data_c = get_mean_and_error(np.array(fpr_c[f'{window}']))
            
            fpr_over_data_q = 1./fpr_data_q[0]
            fpr_over_data_c = 1./fpr_data_c[0]

            fpr_over_error_q = fpr_data_q[1]*(1./np.power(fpr_data_q[0],2)) # sigma_x*(1/x^2)
            fpr_over_error_c = fpr_data_c[1]*(1./np.power(fpr_data_c[0],2))
            
            df_output[f'TPR={window}'] = [f'{fpr_over_data_q:.2f} +/- {fpr_over_error_q:.2f}', f'{fpr_over_data_c:.2f} +/- {fpr_over_error_c:.2f}']
        print(df_output.to_latex(index=False, caption=f'Latent space: {ids[i]}, TPR value +/- {tolerance*100}\%'))
        
    return
        
        
        