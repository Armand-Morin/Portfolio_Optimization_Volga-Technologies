import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sci
import random

# -----------------------------------------------------------------------------------
#                                Paramètres variables
# -----------------------------------------------------------------------------------

# Notionnel total
not_tot = 10**3

# -----------------------------------------------------------------------------------
#                                Import des données
# -----------------------------------------------------------------------------------

list = ["SPY", "QQQ", "VGK", "SCZ", "EWJ", "EEM",  # eq
        "REM", "VNQ", "RWX",  # re
        "TIP", "IEF", "TLT", "BWX",  # fi
        "DBC", "GLD"  # como
        ]

data = yf.download(list, start="2021-01-01", end="2022-02-14")

# -----------------------------------------------------------------------------------
#                              Calcul des log-returns
# -----------------------------------------------------------------------------------

# print(data.loc['2020-12-31':'2021-05-25'])
# print(data.iloc[:100])

data1_St = data.iloc[:100]['Adj Close']
data2_St_moins_1 = data1_St.shift(periods=1)

data3 = data1_St.apply(np.log)
data4 = data2_St_moins_1.apply(np.log)

lg = data3-data4
log_returns = lg[1:]

# Matrice de covariance
df1 = log_returns.cov().copy()
Sigma = df1.to_numpy()

# Matrice de corrélation
df2 = log_returns.corr().copy()
Rho = df2.to_numpy()

"""
print(Sigma)
print(Rho)
"""
# -----------------------------------------------------------------------------------
#                                Target functions
# -----------------------------------------------------------------------------------

# Pour min_var


def target_function_mv(P, Sigma):
    return 10**5*np.dot(np.transpose(P), np.dot(Sigma, P))/(not_tot*not_tot)


def target_function_mv_k(P, Sigma, k):
    return 10**k*target_function_mv(P, Sigma)

# Pour min_correl


def target_function_mc(P, Rho):
    return 10**5*np.dot(np.transpose(P), np.dot(Rho, P))/(not_tot*not_tot)


def target_function_mc_k(P, Rho, k):
    return 10**k*target_function_mc(P, Rho)

# -----------------------------------------------------------------------------------
#                                Optimisation
# -----------------------------------------------------------------------------------

# ---------------------- Méthode 1 : Minimum variance ---------------------------------

# L'algorithme d'optimisation semble donner les meilleurs résultats lorsque la valeur
# de la fonction à son minimum est proche de 1...


def erreurs_mv(solution, soluce):
    err = np.array([0 for i in range(15)])
    for i in range(15):
        err[i] = (np.abs((solution[i]-soluce[i])/soluce[i])*100)
    print("min error : ", np.min(err), "%")
    print("max error : ", np.max(err), "%")
    print("mean error : ", np.mean(err), "%")
    print("error on target function : ", np.abs((target_function_mv(solution, Sigma) -
          target_function_mv(soluce, Sigma))/target_function_mv(soluce, Sigma))*100, "%", "\n")


def min_var(debug=True):  # avec scipy.optimize.minimize

    # Solution analytique du problème fermé

    ones = np.ones((15, 1))
    inv_sig = np.linalg.inv(Sigma)
    prod = np.dot(inv_sig, ones)
    alpha = np.dot(np.transpose(ones), prod)
    solus = prod*not_tot/alpha
    soluce = [x[0] for x in solus]

    # Résolution algorithmique

    L = np.array([random.randint(-10, 10) for i in range(15)])
    if np.sum(L) == 0:
        L[0] += 1
    initial_guess = L*not_tot/np.sum(L)
    contrainte = sci.LinearConstraint(np.ones(15), not_tot, not_tot)
    y = sci.minimize(target_function_mv_k, initial_guess, args=(
        Sigma, 0), method='SLSQP', constraints=(contrainte), options={'maxiter': 300, 'ftol': 1e-09, })
    solution = y.x
    if debug == True:
        print("iterations :", y.nit)
        print("f(x) : ", y.fun)
        erreurs_mv(solution, soluce)
    K = -np.floor(np.log10(y.fun))

    k = 1
    while y.nit > 1 and k < 10:
        initial_guess = solution
        y = sci.minimize(target_function_mv_k, initial_guess, args=(
            Sigma, K), method='SLSQP', constraints=(contrainte), options={'maxiter': 300, 'ftol': 1e-09, })
        solution = y.x
        if debug == True:
            print("iterations :", y.nit)
            print("f(x) : ", y.fun)
            erreurs_mv(solution, soluce)
        K += -np.floor(np.log10(y.fun))
        k += 1

    print(solution)

# ---------------------- Méthode 2 : Minimum correlation ---------------------------------


def erreurs_mc(solution, soluce):
    err = np.array([0 for i in range(15)])
    for i in range(15):
        err[i] = (np.abs((solution[i]-soluce[i])/soluce[i])*100)
    print("min error : ", np.min(err), "%")
    print("max error : ", np.max(err), "%")
    print("mean error : ", np.mean(err), "%")
    print("error on target function : ", np.abs((target_function_mc(solution, Rho) -
          target_function_mc(soluce, Rho))/target_function_mc(soluce, Rho))*100, "%", "\n")


def min_correl(debug=True):  # avec scipy.optimize.minimize

    # Solution analytique du problème fermé

    ones = np.ones((15, 1))
    inv_rho = np.linalg.inv(Rho)
    prod = np.dot(inv_rho, ones)
    alpha = np.dot(np.transpose(ones), prod)
    solus = prod*not_tot/alpha
    soluce = [x[0] for x in solus]

    L = np.array([random.randint(-10, 10) for i in range(15)])
    if np.sum(L) == 0:
        L[0] += 1
    initial_guess = L*not_tot/np.sum(L)
    contrainte = sci.LinearConstraint(np.ones(15), not_tot, not_tot)
    y = sci.minimize(target_function_mc_k, initial_guess, args=(
        Rho, 0), method='SLSQP', constraints=(contrainte), options={'maxiter': 300, 'ftol': 1e-09, })
    solution = y.x
    if debug == True:
        print("iterations :", y.nit)
        print("f(x) : ", y.fun)
        erreurs_mc(solution, soluce)
    K = -np.floor(np.log10(y.fun))

    k = 1
    while y.nit > 1 and k < 10:
        initial_guess = solution
        y = sci.minimize(target_function_mc_k, initial_guess, args=(
            Rho, K), method='SLSQP', constraints=(contrainte), options={'maxiter': 300, 'ftol': 1e-09, })
        solution = y.x
        if debug == True:
            print("iterations :", y.nit)
            print("f(x) : ", y.fun)
            erreurs_mc(solution, soluce)
        K += -np.floor(np.log10(y.fun))
        k += 1

    std_dev = np.array([solution[i]/Sigma[i, i] for i in range(15)])
    beta = np.sum(std_dev)

    for i in range(15):
        solution[i] = std_dev[i]*not_tot/beta

    print(solution)


# min_var()
min_correl()
