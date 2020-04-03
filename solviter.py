# -*- encoding: utf-8 -*-
#  ligne prÃ©cedente pour pouvoir avoir les accents dans les commentaires

import numpy as np
from scipy import sparse
import scipy
from scipy.sparse.linalg import spsolve

pi = np.pi

####  fonction construisant la matrice et le second membre ####
def systeme(ksolve,ksch,N):
    if ksolve <= 4: #matrice ordinaire
        A = np.zeros([N * N, N * N])
    else: #ksolve=5: matrice creuse, le format "lil" est le plus rapide en remplissage
        A = sparse.lil_matrix((N * N, N * N))  # ne marche pas avec []

    #
    # Generation du maillage selon les axes x et y
    #
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    if ksch == 1:    # Remplissage dans le la matrice A dans le cas du schÃ©ma Ã  5 points
        for i in range(0, N):
            A[i, i] = 1

        for i in range(1, N-1):
            ii = i*N
            A[ii, ii] = 1
            for j in range(1, N-1):
                ii = i*N + j
                A[ii, ii] = 1
                A[ii, ii - 1] = -1. / 4.
                A[ii, ii + 1] = -1. / 4.
                A[ii, ii - N] = -1. / 4.
                A[ii, ii + N] = -1. / 4.

            ii = i*N + N - 1
            A[ii, ii] = 1

        for i in range(N*(N-1)-1, N * N):
            A[i, i] = 1

    else:
        # Remplissage de la matrice A dans le cas du schÃ©ma Ã  9 points
        print ("ksch=2 direct non implÃ©mentÃ©")
        A=np.eye(N*N)

    # Remplissage du vecteur b
    b = np.zeros(N * N)

    for i in range(N):
        ii = i*N + N -1
        b[ii] = np.sin(pi * x[i])

    for i in range(N):
        ii = (N - 1) * N + i
        b[ii] = np.sin(pi * y[i])

    return A,b

####  fonction rÃ©solvant le systÃ¨me  ####
def direct(ksolve, ksch , N, Te):  #
    A,b = systeme(ksolve,ksch,N)

    # RÃ©solution du systÃ¨me par mÃ©thode LU
    #matrice ordinaire:
    if ksolve == 4:
        resu = np.linalg.solve(A, b)
    #matrice creuse:
    else:
        resu = scipy.sparse.linalg.spsolve (scipy.sparse.csr_matrix(A), b)

    # Remise en forme de la solution
    T = np.zeros([N, N])
    ii = 0
    for i in range(0,N):
        for j in range(0,N):
            T[i, j] = resu[ii]
            ii = ii + 1

    err = np.linalg.norm(T-Te)/N
    return (err,T)
