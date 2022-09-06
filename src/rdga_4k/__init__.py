def catbird(n, rate, feat_sig, lmbd=.8, eps=.2, random_state=None):

    # libraries
    import numpy as np
    import scipy
    import math
    from numpy.random import RandomState

    if random_state is None:
        random_state = RandomState()
    elif type(random_state) == int:
        random_state = RandomState(random_state)
    else:
        assert type(random_state) == RandomState

    # checking
    assert type(n) == int and n > 1

    assert isinstance(rate, list)
    
    assert isinstance(feat_sig, list) and max(feat_sig) <= n

    assert type(lmbd) == float and (lmbd >= 0.0 and lmbd <= 1.0)

    assert type(eps) == float and (eps >= 0.0 and eps <= 1.0)

    # tools
    def binarize(x, lmbd):
        xbin = []
        for i in range(len(x)):
            if x[i] < lmbd:
                xbin.append(1)
            else:
                xbin.append(0)
        return xbin

    # initialization
    X = []
    y = []
    q = -1

    for i in range(len(rate)):
        W = random_state.normal(0, 1, (feat_sig[i], feat_sig[i]))
        idx = list(random_state.choice(list(range(n)), size=feat_sig[i], replace=False))
        
        q += 1
        for j in range(rate[i]):    
            A = [random_state.normal(0, 1, feat_sig[i])]
            A_W = [[sum(a*b for a,b in zip(A_row,W_col)) for W_col in zip(*W)] for A_row in A]
            A_W_divn = [_/(math.sqrt(feat_sig[i])) for _ in A_W[0]]
            A_W_cum_dist = scipy.stats.norm.cdf(A_W_divn)
            
            A_W_sig_bin = random_state.binomial(1, eps, n)
            A_W_sig_bin[idx] = binarize(A_W_cum_dist, lmbd)

            y.append(q)
            X.append(A_W_sig_bin)

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
    
    return X, y
    
def catbird2(n, rate, feat_sig, lmbd=.8, eps=.2, random_state=None):

    # libraries
    import numpy as np
    import scipy
    import math
    from numpy.random import RandomState

    if random_state is None:
        random_state = RandomState()
    elif type(random_state) == int:
        random_state = RandomState(random_state)
    else:
        assert type(random_state) == RandomState

    # checking
    assert type(n) == int and n > 1

    assert isinstance(rate, list)
    
    assert isinstance(feat_sig, list) and max(feat_sig) <= n

    assert type(lmbd) == float and (lmbd >= 0.0 and lmbd <= 1.0)

    assert type(eps) == float and (eps >= 0.0 and eps <= 1.0)

    # tools
    def binarize(x, lmbd):
        xbin = []
        for i in range(len(x)):
            if x[i] < lmbd:
                xbin.append(1)
            else:
                xbin.append(0)
        return xbin

    # initialization
    X = []
    y = []
    q = -1

    for i in range(len(rate)):
        G = random_state.gamma(shape=2, scale=1, size=(feat_sig[i], feat_sig[i]))
        idx = list(random_state.choice(list(range(n)), size=feat_sig[i], replace=False))
        
        q += 1
        for j in range(rate[i]):    
            U = random_state.uniform(low=0, high=1, size=feat_sig[i])
            E = U@G
            A_W_cum_dist = scipy.stats.erlang.cdf(E, feat_sig[i])
            A_W_sig_bin = random_state.binomial(1, eps, n)
        
        A_W_sig_bin[idx] = binarize(A_W_cum_dist, lmbd)

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
    
    return X, y