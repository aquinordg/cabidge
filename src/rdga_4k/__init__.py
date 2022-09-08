def catbird(n, rate, feat_sig, lmbd=.8, eps=.2, random_state=None):

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