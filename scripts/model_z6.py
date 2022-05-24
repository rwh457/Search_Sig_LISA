import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

z = np.array([1, 4, 7, 10, 13, 16])
c_z = np.array([0, 4, 8, 15, 19, 23])
lw = 1

with plt.style.context(['science', 'nature', 'high-vis']):

    fig = plt.figure(figsize=(3,2.5), dpi=150)
    ax1 = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    axins = ax1.inset_axes([0.3, 0.48, 0.25, 0.25])
    fpr_tpr_auc = np.load('model_z6.npy', allow_pickle=True)
    ax1.plot([0, 1], [0, 1], color='black', lw=lw, linestyle=':', label='Random')    
    for i in range(6):
        fpr, tpr, auc = fpr_tpr_auc[i]
        label = np.where(fpr > 0)
        ax1.plot(fpr, tpr, lw=lw, label=rf'$z = {z[i]}\,($'+ 'AUC'+ rf'$= {auc:0.4f})$', #color=cmap(norm(c_z[i]))
                )
        ax1.legend(borderaxespad=0)

        axins.plot(fpr[label], tpr[label])
    ax1.set_xlim([-0.05, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    
    # sub region of the original image
    axins.set_xscale('log')
    axins.set_xlim(xmin=2e-4,xmax=0.1)
    axins.set_ylim(ymax=1.02)
    ax1.indicate_inset_zoom(axins, edgecolor="black", facecolor='w', ls='--')
    axins.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.savefig('model_z6.pdf', bbox_inches='tight', dpi=300) 
#     fig.savefig('model_z6.pdf', bbox_inches='tight')