# -*- encoding: utf-8 -*-
#  ligne prÃ©cedente pour pouvoir avoir les accents dans les commentaires

import numpy as np
import matplotlib.pyplot as plt
import time as time


def relax(ksolve, ksch, N):
    print(f"ksolve={ksolve} N={N} ksh={ksh}")
    niter = 0  # nombre d'itÃ©ration des mÃ©thodes itÃ©ratives
    nitermax = 100000  # nombre maxi d'iteration de jacobi
    restab = np.zeros(nitermax)
    errtab = np.zeros(nitermax)
    pi = np.pi
    #
    # Generation du maillage selon les axes x et y
    #
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    #
    #    solution approchee  T
    T = np.zeros([N, N])

    #
    om = 2. * (1 - np.pi / N)
    ####solution exacte Te
    Te = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            Te[i, j] = (1 / np.sinh(np.pi)) * (
                        np.sinh(np.pi * x[i]) * np.sin(np.pi * y[j]) + np.sin(np.pi * x[i]) * np.sinh(np.pi * y[j]))
    #####
    #
    #  Application des conditions aux limites (valeurs imposees en debut de calcul)
    T[0, :] = 0.
    T[:, 0] = 0.
    for i in range(0, N):
        T[N - 1, i] = np.sin(pi * y[i])

    for i in range(0, N):
        T[i, N - 1] = np.sin(pi * x[i])

    debtime = time.time()
    res = 1  # rÃ©sidu
    while res > 1.0E-12 and niter < nitermax:  # on continu les iterations tant que le residu est trop grand
        #  stockage du niveau N+1 au niveau N
        U = T.copy()
        if ksolve == 1:  # mÃ©thode de jacobi
            if ksch == 1:
                #  methode de jacobi a 5 Pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = 0.25 * (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1])

            if ksch == 2:
                #  methode de jacobi a 9 pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = 0.2 * (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1]) + 0.05 * (
                                    U[i + 1, j + 1] + U[i - 1, j - 1] + U[i + 1, j - 1] + U[i - 1, j + 1])

        if ksolve == 2:  # mÃ©thode de Gauss Seidel
            if ksch == 1:
                #  methode de Gauss Seidel a 5 Pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = 0.25 * (U[i + 1, j] + T[i - 1, j] + U[i, j + 1] + T[i, j - 1])
            if ksch == 2:
                #  methode de Gauss Seidel a 9 pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = 0.2 * (U[i + 1, j] + T[i - 1, j] + U[i, j + 1] + T[i, j - 1]) + 0.05 * (
                                    U[i + 1, j + 1] + T[i - 1, j - 1] + T[i + 1, j - 1] + T[i - 1, j + 1])

        if ksolve == 3:  # mÃ©thode SOR
            if ksch == 1:
                #  methode SOR a 5 Pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = U[i, j] * (1 - om) + om * 0.25 * (
                                    T[i - 1, j] + U[i + 1, j] + T[i, j - 1] + U[i, j + 1])

            if ksch == 2:
                #  methode SOR a 9 pts
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        T[i, j] = U[i, j] * (1 - om) + om * (
                                    0.2 * (U[i + 1, j] + T[i - 1, j] + U[i, j + 1] + T[i, j - 1]) + 0.05 * (
                                        U[i + 1, j + 1] + T[i - 1, j - 1] + T[i + 1, j - 1] + T[i - 1, j + 1]))

        niter = niter + 1

        # Calcul du residu en norme L2
        res = np.linalg.norm(T - U) / N
        restab[niter - 1] = np.log10(res)

        # calcul de l'erreur en norme L2
        err = np.linalg.norm(T - Te) / N
        errtab[niter - 1] = np.log10(err)

    fintime = time.time()
    return (niter, errtab, restab, T, err, fintime - debtime)


'''
ni,errt,res,t,err,temp=relax(3, 1, 60, Te)
print(f"le nombre d'iteration est {ni} \nle log10 de l'erreur est {err}")
print("temps cpu", fintime-debtime)
print("temps cpu2", temp)
'''
#####graphique
import matplotlib.pyplot as plt
from scipy import stats

Tc = [6.02, 48.42, 163.84, 389.35]
# Nc=[-3.15,-3.77,-4.16]
Nc = [20, 40, 60, 80]
#Nc = [5,10,20]
nis = []
temps = []
errs = []
ksh = 2
for ksolve in [1, 2, 3]:
    nis.append([])
    temps.append([])
    errs.append([])
    for N in Nc:
        ni, errt, res, T, err, temp = relax(ksolve, ksh, N)
        nis[ksolve - 1].append(ni)
        temps[ksolve - 1].append(temp)
        errs[ksolve - 1].append(err)
print("nis", nis)
print("errs", errs)
print("temps", temps)
coul = ['b', 'r', 'g']
noms = ['Jacobi', 'Gauss-Seidel', 'SOR']

for ksolve in [1, 2, 3]:
    # slope, intercept, r_value, p_value, std_err = stats.linregress(Nc, temps[ksolve-1])
    plt.plot(Nc, temps[ksolve - 1], f"{coul[ksolve - 1]}-x", label=f"méthode={noms[ksolve - 1]}")
    # print(f"L'équation de la droite régression est 't={intercept:8.2f} + {slope:8.2f}N'")
plt.legend()
plt.title(f"ksh={ksh}")
plt.xlabel('N')
plt.ylabel('Temps de calcul (s)')
plt.savefig(f"NvsT_{ksh}.png")
plt.show()

for ksolve in [1, 2, 3]:
    # slope, intercept, r_value, p_value, std_err = stats.linregress(Nc, temps[ksolve-1])
    plt.plot(temps[ksolve - 1], errs[ksolve - 1], f"{coul[ksolve - 1]}-x", label=f"méthode={noms[ksolve - 1]}")
    #lprint(f"L'équation de la droite régression est 't={intercept:8.2f} + {slope:8.2f}N'")
plt.legend()
plt.title(f"ksh={ksh}")
plt.ylabel('Erreur')
plt.xlabel('Temps de calcul (s)')
plt.savefig(f"TvsE_{ksh}.png")
plt.show()

