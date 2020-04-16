import numpy as np

def indsAndShifts(dim =3, k=2):
    s0 = [-1, 1]
    s=[]
    ind =[]
    if k==1:
        for i0 in range(dim):
            ind.append([i0])
        for i0 in s0:
            s.append([i0])
    elif k==2:
        for i0 in range(dim):
            for i1 in range(i0+1,dim):
                    ind.append([i0,i1])
        for i0 in s0:
            for i1 in s0:
                s.append([i0, i1])
    elif k==3:
        for i0 in range(dim):
            for i1 in range(i0+1,dim):
                for i2 in range(i1 + 1, dim):
                    ind.append([i0,i1,i2])
        for i0 in s0:
            for i1 in s0:
                for i2 in s0:
                    s.append([i0,i1,i2])
    elif k==4:
        for i0 in range(dim):
            for i1 in range(i0+1,dim):
                for i2 in range(i1 + 1, dim):
                    for i3 in range(i2 + 1, dim):
                        ind.append([i0,i1,i2,i3])
        for i0 in s0:
            for i1 in s0:
                for i2 in s0:
                    for i3 in s0:
                        s.append([i0,i1,i2,i3])
    else:
        raise ValueError(f"Noe written k={k}")
    return ind, s

#=================================================================#
if __name__ == '__main__':
    ind, s = indsAndShifts(dim=3, k=2)
    print(f"ind={ind} s={s}")
