import numpy as np
import pandas as pd
import math
import random

def cabidge(lmbd, eps, m, n, k, seed):

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def sig_f(x):
        return [sigmoid(i) for i in x]

    def binarize(x, lmbd):
        xbin = []
        for i in range(len(x)):
            if x[i] < lmbd:
                xbin.append(1)
            else:
                xbin.append(0)
        return xbin

    def sample_idx(list_, n, ref):
        idx = random.sample(list_, n)
        idx.sort()
        while idx in ref:
            idx = random.sample(list_, n)
            idx.sort()
        return idx

    n_W = int(n//2)+1
    n_W_res = n - n_W

    n_samples_per_center = [int(m // k)] * k

    for i in range(m % k):
        n_samples_per_center[i] += 1

    X = []
    y = []
    idx_used = []
    k = -1

    for i in n_samples_per_center:
        np.random.seed(seed)
        W = np.random.normal(0, 1, (n_W, n_W))
        idx = sample_idx(list(range(n)), n_W_res, idx_used, seed)
        
        k += 1
        for j in range(i):
            np.random.seed(seed)
            A = [np.random.normal(0, 1, n_W)]
            A_W = [[sum(a*b for a,b in zip(A_row,W_col)) for W_col in zip(*W)] for A_row in A]
            A_W_sig = sig_f(A_W[0])
            
            for l in idx:
                A_W_sig.insert(l, eps)
            
            A_W_sig_bin = binarize(A_W_sig, lmbd)
            
            y.append(k)
            X.append(A_W_sig_bin)
        
        idx_used.append(idx)
        
    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
    
    return X, y

