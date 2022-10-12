import numpy as np
import pandas as pd
import math
from numpy.random import RandomState
from scipy.stats import norm

def get_rate(N, k, n_min):
    """
    Parameters:
    `N` int > 1: approximate number of examples
    `k` int > 1: number of clusters
    `n_min` int: minimum number of examples per cluster
    """
    
    assert type(N) == int and N > 1
    assert type(k) == int and k > 1
    
    rate_c = []
    resto = N
    for j in range(2, k+2):
        rate_c.append(int(resto/j))
        resto = N - sum(rate_c)
    rate_s = [int(sum(rate_c)/k) for i in range(k)]
    rate=[rate_s, rate_c]
    
    assert (min(rate_s) and min(rate_c)) >= n_min
    
    return(rate)

def catbird(n_feat, feat_sig, rate, lmbd=.8, eps=.2, random_state=None):
    """
    Parameters:
    `n_feat` int > 1: number of features
    `feat_sig` list: number of significant features (no noise), equal to the size of the rate list 
    `rate` list: division of examples into clusters, equal to the size of the feat_sig list
    `lmbd` float 0,1: intersection factor between features
    `eps` float 0,1: feature noise rate
    `random_state` int: random seed
    """

    if random_state is None:
        random_state = RandomState()
    elif type(random_state) == int:
        random_state = RandomState(random_state)
    else:
        assert type(random_state) == RandomState

    # checking
    assert type(n_feat) == int and n_feat > 1

    assert isinstance(rate, list)

    assert isinstance(feat_sig, list) and max(feat_sig) <= n_feat

    assert type(lmbd) == float and (lmbd >= 0.0 and lmbd <= 1.0)

    assert type(eps) == float and (eps >= 0.0 and eps <= 1.0)

    # tools
    discretize = np.vectorize(lambda x, thr: 1 if x < thr else 0)

    # initialization
    X = []
    y = []
    q = -1

    for i in range(len(rate)):
        W = random_state.normal(0, 1, (feat_sig[i], feat_sig[i]))
        idx = random_state.choice(n_feat, size=feat_sig[i], replace=False)

        q += 1
        for j in range(rate[i]):
            A = random_state.normal(0, 1, (1, feat_sig[i]))
            A_W = A @ W
            A_W_norm = norm.cdf(A_W / math.sqrt(feat_sig[i]))

            result = random_state.binomial(1, eps, n_feat)
            result[idx] = discretize(A_W_norm, lmbd)

            y.append(q)
            X.append(result)

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)

    return X, y
    
def canard(n_feat, n_cat, rate, lmbd=10, eps=.3, random_state=None):
    """
    Parameters:
    `n_feat` int > 1: number of features
    `n_cat` int > 1: number of categories
    `rate` list: division of examples into clusters
    `lmbd` int >= 1: intersection factor between features
    `eps` float 0,1: feature noise rate
    `random_state` int: random seed
    """
    
    if random_state is None:
        random_state = RandomState()
    elif type(random_state) == int:
        random_state = RandomState(random_state)
    else:
        assert type(random_state) == RandomState
        
    # checking
    assert type(n_feat) == int and n_feat > 1
    
    assert type(n_cat) == int and n_cat > 1

    assert isinstance(rate, list)

    assert type(lmbd) == int and lmbd >= 1 

    assert type(eps) == float and (eps >= 0.0 and eps <= 1.0)

    # initialization
    X = []
    y = []
    q = -1
    cats = list(range(n_cat))
    bins = [i / n_cat for i in range(0, (n_cat+1))]

    for m in rate:
        q += 1
        a = n_feat*[1]
        b = n_feat*[1]
        idx = list(random_state.choice(int(n_feat), size=int(n_feat*(1-eps)), replace=False))

        for i in idx:
            a[i] = 1 + lmbd * random_state.random_sample()
            b[i] = 1 + lmbd * random_state.random_sample()

        for _ in range(m):
            ex = np.array([ pd.cut(random_state.beta(a[j], b[j], 1), bins = bins, labels=cats)[0] for j in range(n_feat) ])
            X.append(ex)
            y.append(q)
            
    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
            
    return X, y
