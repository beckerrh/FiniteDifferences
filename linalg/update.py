import numpy as np
import scipy as sp

class Update:
    def __init__(self, matrix):
        self.matrix = matrix
        # print(f"np.shares_memory(matrix,self.matrix) = {np.shares_memory(matrix.data,self.matrix.data)}")
        n = matrix.shape[0]
        self.r, self.Aw = np.zeros(n), np.zeros_like(n)
        self.omegamax, self.omegamin = 1000, 0.1
    def update(self, u, w, f, r=None):
        """
        min |f- A(u+omega*w)| --> omega = 2 * r.T*Aw / |Aw|^2
        w.T *( f- A(u+omega*w)) --> omega = w.T *r / w.T*Aw
        """
        smooth = "*"
        Aw, matrix = self.Aw, self.matrix
        if r is None:
            r = self.r
            r[:] = f[:] - matrix.dot(u)
            smooth = ""
        resold2 = r.dot(r)
        Aw = matrix.dot(w)
        omega2 = r.dot(Aw)/Aw.dot(Aw)
        omega = w.dot(r)/w.dot(Aw)
        omega = max(min(omega,self.omegamax),self.omegamin)
        resnew2 = resold2 - 2*omega*r.dot(Aw) + omega**2*Aw.dot(Aw)
        # print(f"{smooth:2s}omega = {omega:6.2f} omega2 = {omega2:6.2f} resnew2/resold2={resnew2/resold2:5.2f} ")
        u += omega*w
        r -= omega*Aw
        return u
