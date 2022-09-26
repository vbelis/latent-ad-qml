import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import sklearn.metrics as skl
import os
import mplhep as hep
import pathlib

import anpofah.model_analysis.roc_analysis as ra
import util.logging as log
import pofah.jet_sample as jesa

sig_name_dict = {
    
    'GtoWW15br' : r'Broad $G \to \, WW \,\, 1.5 \, TeV$', 
    'GtoWW35na' : r'Narrow $G \to \, WW \,\, 3.5 \, TeV$', 
    'AtoHZ35' : r'$A \to \, HZ \to \, ZZZ \,\, 3.5 \, TeV$'
}

def prepare_labels_and_losses_signal_comparison(qcd_losses, sig_losses):

    # ******
    #   qcd_losses : N x 2 (N events x (classic, quantum))
    #   sig_losses : M x N x 2 ( M signals x N events x (classic, quantum))
    # ******

    class_labels = []
    losses = []

    for pos_class_loss in sig_losses: # creates interleaved classic / quantum results
        class_label_arr, loss_arr = ra.get_label_and_score_arrays(qcd_losses, pos_class_loss) # stack losses and create according labels per strategy
        class_labels.append(class_label_arr)
        losses.append(loss_arr)

    return class_labels, losses

def prepare_labels_and_losses_train_sz_comparison(qcd_losses, sig_losses):

    # ******
    #   qcd_losses : N x 2 (N events x (classic, quantum))
    #   sig_losses : M x N x 2 ( M signals x N events x (classic, quantum))
    # ******

    class_labels = []
    losses = []

    for neg_class_loss, pos_class_loss in zip(qcd_losses, sig_losses): # creates interleaved classic / quantum results
        class_label_arr, loss_arr = ra.get_label_and_score_arrays(neg_class_loss, pos_class_loss) # stack losses and create according labels per strategy
        class_labels.append(class_label_arr)
        losses.append(loss_arr)

    return class_labels, losses




def plot_roc(class_labels, losses, legend_colors, legend_colors_title, test_n=int(1e4), title='ROC', legend_loc='best', plot_name='ROC', fig_dir=None, x_lim=None, log_x=True, fig_format='.png'):

    # ******
    #   class labels : K x 2 x N (K signals/variations x (classic, quantum) x N events)
    #   sig_losses : K x 2 x N (K signals/variations x (classic, quantum) x N events)
    # ******


    plt.style.use(hep.style.CMS)

    line_type_n = len(class_labels[0]) # classic vs quantum

    palette = ['#3E96A1', '#EC4E20', '#FF9505', '#713E5A']
    styles = ['solid', 'dashed', 'dotted'][:line_type_n]# 2 styles for classic vs quantum times number of signals

    aucs = []
    fig = plt.figure(figsize=(8, 8)) # figsize=(5, 5)

    # import ipdb; ipdb.set_trace()

    for y_true_qc, loss_qc, color in zip(class_labels, losses, palette):
        for y_true, loss, style in zip(y_true_qc, loss_qc, styles): 
            fpr, tpr, threshold = skl.roc_curve(y_true, loss)
            aucs.append(skl.roc_auc_score(y_true, loss))
            if log_x:
                plt.loglog(tpr, 1./fpr, linestyle=style, color=color) # label=label + " (auc " + "{0:.3f}".format(aucs[-1]) + ")",
            else:
                plt.semilogy(tpr, 1./fpr, linestyle=style, color=color)

    dummy_res_lines = [Line2D([0,1],[0,1],linestyle=s, color='gray') for s in styles[:2]]
    plt.semilogy(np.linspace(0, 1, num=test_n), 1./np.linspace(0, 1, num=test_n), linewidth=1.2, linestyle='solid', color='silver')
    
    # add 2 legends (vae score types and resonance types)
    lines = plt.gca().get_lines()
    legend1 = plt.legend(dummy_res_lines, [r'Quantum', r'Classical'], loc='lower left', frameon=False, title='algorithm', \
            handlelength=1.5, fontsize=14, title_fontsize=17, bbox_to_anchor=(0,0.28))
    legend_colors = [l + " (auc " + "{0:.3f} , {1:.3f}".format(aucs[i*2],aucs[i*2+1]) + ")" for i,l in enumerate(legend_colors)]
    legend2 = plt.legend([lines[i*line_type_n] for i in range(len(legend_colors))], legend_colors, loc='lower left', \
            frameon=False, title=legend_colors_title, fontsize=14, title_fontsize=17)
    legend1._legend_box.align = "left"
    legend2._legend_box.align = "left"
    for leg in legend1.legendHandles:
        leg.set_linewidth(2.2)
        leg.set_color('gray')
    for leg in legend2.legendHandles:
        leg.set_linewidth(2.2) 
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    plt.title(title , loc="right",fontsize=16)

    plt.grid()
    if x_lim:
        plt.xlim(left=x_lim)
    plt.xlabel('True positive rate',fontsize=16)
    plt.ylabel('1 / False positive rate',fontsize=16)
    plt.tight_layout()
    if fig_dir:
        print('writing ROC plot to {}'.format(fig_dir))
        fig.savefig(os.path.join(fig_dir, plot_name + fig_format), bbox_inches='tight')
    plt.close(fig)
    return aucs
