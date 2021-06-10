import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Given a matrix, this method shows it so that larger values have lighter colors
# If sim == true, it only shows the lower triangular matrix (for symmetric matrices).
# If abs == true, cells are colored by absolute value (for correlation matrix, between -1 and 1)
def show(df,sim=True,cmap=None,annot=True,corr=True):
    ax = plt.subplots(figsize=(40,20))[1]
    mask=None
    if sim:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    dfaux = df
    if corr:
        dfaux = abs(dfaux)
    hm = sns.heatmap(dfaux, mask=mask, cmap=cmap, annot=df, ax=ax, linewidths=1, linecolor='black')
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)
    plt.show(hm)

# Given a set p of lists of p-values, it shows the distribution of each pair of lists
def distrib_pv(p,s,names):
    npt = np.array(p).transpose()
    fig = plt.figure()
    gs = fig.add_gridspec(s, s, hspace=0.1, wspace=0.1)
    axs = gs.subplots(sharex='col', sharey='row')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for i in range(s):
        axs[i, 0].set(ylabel=names[i])
        for j in range(0,i):
            axs[i, j].plot(npt[i], npt[j])
    for j in range(0,s):
        axs[s-1, j].set(xlabel=names[j])