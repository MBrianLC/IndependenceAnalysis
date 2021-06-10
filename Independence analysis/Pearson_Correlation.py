import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats

# Divides a set u of n sequences into div parts and calculates in each one the Pearson correlation and the p-value
# associated. It returns the mean of the p-values and the result of applying a Kolmogorov-Smirnov test to them.
def corrsig(u,div=100):
    l = len(u)
    d = l//div
    df = pd.DataFrame(u[0:d])
    corr = df.corr()
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr.shape)
    pks = [[[pval[i][j]] for j in range(len(pval[i]))] for i in range(len(pval))]
    for i in range (1,div):
        df = pd.DataFrame(u[d*i:d*(i+1)])
        corr = df.corr()
        aux = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr.shape)
        pval += aux
        pks = [[pks[i][j]+[aux[i][j]] for j in range(len(aux[i]))] for i in range(len(aux))]
    val = [[pval[i][j]/div for j in range(len(pval[i]))] for i in range(len(pval))]
    ks = [[stats.kstest(pks[i][j],'uniform')[1] for j in range(len(pks[i]))] for i in range(len(pks))]
    return (val,ks)

# Shows the correlation matrix
def showCorrelation(df):
    Utils.show(df.corr())

v = [0.01,0.05,0.1]
def significance(x,v):
    cont = 0
    aux = 1/len(v)
    for t in v:
        if x<=t:
            cont = cont+aux
    return cont
	
# Returns the matrix of p-values of the correlation
def showSignificance(df,show_m=True):
    corr = df.corr()
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr.shape)
    if not show_m:
        pval = pval.applymap(lambda x: significance(x,v))
    Utils.show(pval,True,sns.cubehelix_palette(start=2.5, rot=0, dark=0.2, light=0.8, reverse=False, as_cmap=True),show_m)
