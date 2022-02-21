import numpy as np
import yfinance as yf
import pandas as pd
from math import pi, exp, sqrt
import scipy.stats as ss

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


def generate_return_series(prices):
    """Compute daily return series for given price series"""
    returns = np.zeros(len(prices) - 1)
    for i in range(len(prices) - 1):
        day_return = (prices[i + 1] - prices[i]) / prices[i]
        returns[i] = day_return

    return returns


def MaxSR_PTF(S, mu):
    Sinv = np.linalg.inv(S)
    N = S.shape[0]
    return (Sinv @ mu)/(np.ones(N) @ Sinv @ mu)


def Min_Var_PTF(S):
    w = np.linalg.pinv(S).sum(axis=1)
    return w/w.sum()


def EV_PTF_positive(R, mu, m):
    N = R.shape[0]

    #Aw = b
    A = np.ones((2, N))
    A[1] = mu

    r = solvers.qp(matrix(R),  # min w P w
                   matrix(np.zeros(N)),  # min w q
                   A=matrix(A),
                   b=matrix(np.array([1., m])),
                   G=matrix(-np.identity(N)),  # w < h
                   h=matrix(np.zeros(N)))

    return np.array(r['x']).T[0]


def normdist(x, m, sigma):
    return((1/(sqrt(2*pi)*sigma)) * (exp(-0.5*((((x-m)/sigma)**2)))))


def minCorr_init(S):
    """Provides a vector of initial weights for a min correlation potofolio
    heuristic method from David Varadi
    S is a dataframe of real numbers"""
    # step 1
    corrS = np.corrcoef(S)
    N = np.shape(corrS)[0]
    values_corr = corrS[np.triu_indices(N, 1)]

    tri_corr = np.zeros((N, N))

    tri_corr[np.triu_indices(N, 1)] = values_corr
    tri_corr[np.tril_indices(N, -1)] = values_corr

    # step 2
    mu_ro = np.mean(values_corr)
    sigma_ro = np.std(values_corr, axis=1)

    # step 3
    corrA = np.zeros(N)
    for i, j in range(N):
        x = tri_corr[i, j]
        corrA[i, j] = 1-normdist(x, mu_ro, sigma_ro)

    # step 4
    weightsT = np.mean(corrA, axis=0)

    # step 5
    RANK = np.array(ss.rankdata(weightsT))
    weightsRANK = RANK/np.sum(RANK)

    # step 6
    weights = weightsRANK @ corrA
    return(weights/np.sum(weights))
