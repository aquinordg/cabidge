def catbird(m, n, k, lmbd=.5, eps=.5, random_state=None):

    # libraries
    import numpy as np
    import math
    from numpy.random import RandomState

    if random_state is None:
        random_state = RandomState()
    elif type(random_state) == int:
        random_state = RandomState(random_state)
    else:
        assert type(random_state) == RandomState

    # checking
    assert type(m) == int and m > 1

    assert type(n) == int and n > 1

    assert type(k) == int and k > 1

    assert type(lmbd) == float and (lmbd >= 0.0 and lmbd <= 1.0)

    assert type(eps) == float and (eps >= 0.0 and eps <= 1.0)

    # tools
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

    # initialization
    s = int(n//2)+1
    rem = n - s

    n_samples_per_center = [int(m // k)] * k
    for i in range(m % k):
        n_samples_per_center[i] += 1

    X = []
    y = []
    q = -1

    for i in range(len(n_samples_per_center)):
        W = random_state.normal(0, 1, (s, s))
        idx = list(random_state.choice(list(range(n)), size=rem, replace=False))
        idx.sort()

        q += 1
        for j in range(n_samples_per_center[i]):
            A = [random_state.normal(0, 1, s)]
            A_W = [[sum(a*b for a,b in zip(A_row,W_col)) for W_col in zip(*W)] for A_row in A]
            A_W_sig = sig_f(A_W[0])

            for l in idx:
                A_W_sig.insert(l, eps)

            A_W_sig_bin = binarize(A_W_sig, lmbd)

            y.append(q)
            X.append(A_W_sig_bin)

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)

    return X, y

def free_catbird(n, rate, feat_sig, lmbd=.5, eps=.5, random_state=None):

    # libraries
    import numpy as np
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
            A_W_sig = sig_f(A_W[0])

            noise_list = [eps for i in range(n)]

            for l in range(len(A_W_sig)):
                noise_list[idx[l]] = A_W_sig[l]

            A_W_sig_bin = binarize(noise_list, lmbd)

            y.append(q)
            X.append(A_W_sig_bin)

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)

    return X, y

def new_free_catbird(n, rate, feat_sig, lmbd=.5, eps=.5, random_state=None):

    # libraries
    import numpy as np
    import math
    from numpy.random import RandomState
    from scipy.stats import norm

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
    discretize = np.vectorize(lambda x, thr: 1 if x < thr else 0)

    # initialization
    X = []
    y = []
    q = -1

    for i in range(len(rate)):
        W = random_state.normal(0, 1, (feat_sig[i], feat_sig[i]))
        idx = random_state.choice(n, size=feat_sig[i], replace=False)

        q += 1
        for j in range(rate[i]):
            A = random_state.normal(0, 1, (1, feat_sig[i]))
            A_W = A @ W
            A_W_norm = norm.cdf(A_W / math.sqrt(feat_sig[i]))

            result = random_state.binomial(1, eps, n)
            result[idx] = discretize(A_W_norm, lmbd)

            y.append(q)
            X.append(result)

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)

    return X, y


def reduce_nominal(X, n):
    """
    Parameters:
    `X`: output X from the catbird function
    `n`: number of resulting nominal features
    """

    assert X.shape[1] % n == 0

    shift = X.shape[1] // n
    Xnew = np.zeros(shape=(X.shape[0], n))

    for i in range(n):
        Xnew[:, i] = X[:, i*shift:(i*shift + shift)].sum(axis=1)

    return Xnew
