import numpy as np
import matplotlib
import matplotlib.pyplot as plt
auc_list = np.load('imr_auc.npy')

z = np.array([1, 4, 7, 10, 13, 16])
markersize = 4
auc_index = [0,1,2,3]
marker = ['o', 's', 'v', 'D', 'p']
label = ['IMRPhenomD', 'SEOBNRv4', 'SEOBNRE', 'SEOBNRv4P']
with plt.style.context(['science', 'nature', #'retro'
                       ]):
    plt.figure(figsize=(4,3), dpi=150)
    for i, m,l in zip(auc_index, marker, label):
        plt.plot(z, auc_list[i, :], marker=m, label=l, markersize=markersize,alpha=0.8# linewidth=1.2,
                )
#     plt.xlabel(r'$z$')
    plt.xlabel('Redshift')
    plt.ylabel('AUC')
    plt.legend()
    plt.xticks([2, 4, 6, 8,10,12,14,16, ])
plt.savefig('imr_auc.pdf', bbox_inches='tight', dpi=300) 