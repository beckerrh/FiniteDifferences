import numpy as np
import scipy as sp

class GaussSeidel:
    def __init__(self, A, omega=0.8):
        self.A, self.omega, self.n = A, omega, A.shape[0]
        self.data = A.data
        self.offsets = A.offsets
        self.Dinv = 1/A.diagonal()
        self.h = np.empty(self.n)
        self.r = np.empty(self.n)
        self.s = np.empty(self.n)

    def matvec(self, x):
        h, r, s = self.h, self.r, self.s
        h = x*self.Dinv
        s = x - self.A.dot(h)
        r[:] = x[:]
        n_row, n_col, L = self.n, self.n, self.n
        for ii in range(len(self.offsets)):
            k = self.offsets[ii]
            i_start = max(0,-k)
            j_start = max(0,k)
            j_end = min(min(n_row + k, n_col),L)
            N = j_end - j_start
            r[i_start:i_start+N] -= self.data[ii, j_start:j_start+N] * h[j_start:j_start+N]


            # if of >=0:
            #     for i in range(0,self.n-of):
            #         r[i] -= self.data[ii,i+of]*h[i+of]
            # else:
            #     for i in range(-of, self.n):
            #         r[i] -= self.data[ii, i+of] * h[i + of]
        if np.linalg.norm(r-s) > 1e-14:
            raise ValueError(f"error {np.linalg.norm(r-s)}\ns=\n{s}\nr=\n{r}\nr-s=\n{r-s}")
        self.h = self.omega*x * self.Dinv
        return self.h
"""
    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T * diag = diags + (npy_intp)i*L + j_start;
        const T * x = Xx + j_start;
              T * y = Yx + i_start;

        for(I n = 0; n < N; n++){
            y[n] += diag[n] * x[n]; 
        }
    }
"""