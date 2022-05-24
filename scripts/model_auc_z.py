import numpy as np
import matplotlib.pyplot as plt

auc = np.load('model_auc_z.npy')
x = np.array([1, 4, 7, 10, 13, 16])
marker = ['o', 's', 'v', 'D', 'p']
markersize = 4
with plt.style.context(['science', 'nature', #'retro'
                       ]):
    fig = plt.figure(figsize=(4,3), dpi=150)
    for i, (z, m) in enumerate(zip([1,3,6,10,15], marker)):
        plt.plot(x, auc[i, :], marker=m,
                 label=rf'$z={z}$', markersize=markersize)
    plt.xlabel('Redshift')
    plt.ylabel('AUC')
    plt.legend()
    plt.xticks([2, 4, 6, 8,10,12,14,16, ])
    plt.tight_layout();
plt.savefig('model_auc_z.pdf', bbox_inches='tight', dpi=300) 
# fig.savefig('model_auc_z.pdf')