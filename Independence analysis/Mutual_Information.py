import random
import numpy as np
from pyitlib import discrete_random_variable as drv

# Transforms a set p of lists of p-values into a set of lists of integers between 0 and 9
# (each element p ' becomes the integer part of 10 * p')
def discretization(p):
    return (np.array(p)*10).astype(int)%10

# Returns the mutual information of two random samples X and Y, using est as the estimator of probability
def mutual_information(X,Y,est):
    return drv.entropy(Y,estimator=est) - drv.entropy_conditional(Y,X,estimator=est)

# Given a set p of lists of p-values, it returns the mutual information matrix (lower triangular matrix)
def mi_matrix_v1(p,est):
    paux = discretization(p).T
    l = len(paux)
    m = [[0 for j in range(l)] for i in range(l)]
    for i in range(l):
        e = drv.entropy(paux[i],estimator=est)
        for j in range(i,l):
            m[i][j] = mutual_information(paux[i],paux[j],est)/e
    return m

# Given a set p of lists of p-values and an integer q, it returns the mutual information matrix and the matrix
# of the associated p-values through the permutation test (q permutations). Since the matrices are symmetric, they
# are returned as arrays (each element is a cell of the lower triangular matrix)
def permutation_test_v1(p,q,est):
    v = [i for i in range(len(p))]
    paux = discretization(p).T
    paux2 = paux
    l = len(paux)
    comp = [0 for i in range(l*(l-1)//2)]
    mi = []
    e = [0 for i in range(l)]
    for i in range(l):
        e[i] = drv.entropy(paux[i],estimator=est)
    for i in range(l):
        for j in range(i+1,l):
            mi.append(mutual_information(paux[i],paux[j],est)/e[i])
    for k in range(q):
        random.shuffle(v)
        paux2 = paux2[:,v]
        aux = 0
        for i in range(l):
            for j in range(i+1,l):
                if(mutual_information(paux[i],paux2[j],est)/e[i] >= mi[aux]):
                    comp[aux] += 1
                aux += 1
    return (mi,np.array(comp)/q)

# Computes the entropy of all the lists in the set p, as well as the joint entropy of each pair of lists. In
# this case, we estimate the probability by the frequency of appearance of each of the possible values that can
# be obtained by discretizing p
def prob(p,d):
    ent = []
    entj = []
    l = len(p)
    laux = len(p[0])
    for i in range(l):
        f = np.bincount(p[i])/laux
        e = 0
        for n in range(d):
            if (f[n] > 0):
                e -= f[n]*np.log2(f[n])
        ent.append(e)
    for i in range(l):
        p1 = p[i]*d
        for j in range(i+1,l):
            p2 = p1+p[j]
            f = np.bincount(p2)/laux
            e = 0
            for n in range(len(f)):
                if (f[n] > 0):
                    e -= f[n]*np.log2(f[n])
            entj.append(e)
    return ent,entj
    
# An optimized version of the calculation of the mutual information matrix, taking as the estimator of probability
# the frequency of the possible values after the discretization of p
def mi_matrix_v2(p,d = 10):
    paux = discretization(p,d).T
    ent,entj = prob(paux,d)
    l = len(paux)
    m = [[0 for j in range(l)] for i in range(l)]
    aux = 0
    for i in range(l):
        m[i][i] = 1
        for j in range(i+1,l):
            m[i][j] = (ent[i]+ent[j]-entj[aux])/ent[i]
            aux += 1
    return m

# An optimized version of the permutation test, taking as the estimator of probability the frequency of the
# possible values after the discretization of p
def permutation_test_v2(p,q,d = 10):
    v = [i for i in range(len(p))]
    paux = discretization(p,d).T
    paux2 = paux
    ent,entj = prob(paux,d)
    l = len(paux)
    comp = [0 for i in range(l*(l-1)//2)]
    mi = []
    e = [0 for i in range(l)]
    aux = 0
    for i in range(l):
        for j in range(i+1,l):
            mi.append((ent[i]+ent[j]-entj[aux])/ent[i])
            aux += 1
    for k in range(q):
        random.shuffle(v)
        paux2 = paux2[:,v]
        aux = 0
        for i in range(l):
            p1 = paux[i]*d
            for j in range(i+1,l):
                p2 = p1+paux2[j]
                f = np.bincount(p2)/len(p)
                e = 0
                for n in range(len(f)):
                    if (f[n] > 0):
                        e -= f[n]*np.log2(f[n])
                miv = (ent[i]+ent[j]-e)/ent[i]
                if(miv >= mi[aux]):
                    comp[aux] += 1
                aux += 1
    return (mi,np.array(comp)/q)

# Divides a set u of n sequences into div parts and calculates in each one the mutual information and
# permutation test matrices (as arrays). It returns the div arrays obtained
def MI_test(pv,q,div):
    l = len(pv)
    r = l // div
    s = [[] for i in range(div)]
    p = [[] for i in range(div)]
    for i in range(div):
        (s[i], p[i]) = permutation_test(pv[r*i:r*(i+1)],q)
    return (s,p)

# Given the set ts of lists of p-values ts, it finds the mean and a K-S test on each list. It returns the values
# obtained (set of means and K-S test results) displayed as matrices
def MI_sign(ts,size,nTests):
    mmean = [[] for i in range(nTests)]
    mks = [[] for i in range(nTests)]
    cont = 0
    for i in range(nTests):
        mi = []
        mk = []
        for j in range(0,i+1):
            mi.append(0)
            mk.append(0)
        for j in range(i+1,nTests):
            mi.append(ssum(ts[cont])/size)
            mk.append(stats.kstest(ts[cont],'uniform')[1])
            cont += 1
        mmean[i] = mi
        mks[i] = mk
    return (np.array(mmean).transpose(),np.array(mks).transpose())