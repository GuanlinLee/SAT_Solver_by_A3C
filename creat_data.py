from time import time
from random import randint,seed
import pycosat
import numpy as np
seed(time())

def creat_cnf(size_of_X,size_of_C,k_sat):
    size_of_X = size_of_X
    size_of_C = size_of_C
    k_sat = k_sat

    C = []

    for i in range(size_of_C):
        ci = []
        for k in range(k_sat):
            if_not = randint(0, 1)
            elem = randint(1, size_of_X)
            if if_not:
                ci.append(-elem)
            else:
                ci.append(elem)
        C.append(ci)
    #print(C)

    C_np = []

    #if pycosat.solve(C) == 'UNSAT':
    #    print('its not a SAT')
    #else:
        #print(pycosat.solve(C))
    for i in range(size_of_C):
        ci = C[i]
        ci_np = np.zeros(2 * k_sat, int)
        for x in range(k_sat):
            ci_np[x] = abs(ci[x])
            if ci[x] < 0:
                ci_np[k_sat + x] = 1
        C_np.append(list(ci_np))
    C_np = np.array(C_np)
    C_np.shape = (size_of_C, 2 * k_sat)
    #print(C_np)

    X = []
    for i in range(1, size_of_X + 1):
        X.append(randint(0, 1))
    #print(X)
    X=np.array(X)

    return X.astype(np.float32),C_np.astype(np.float32)

'''
acc=[]

for i in range(size_of_C):
    x_list=[]
    not_list=[]
    ci_np=C_np[i]
    for k in range(k_sat):
        x_list.append(X[ci_np[k]-1])
        if ci_np[k_sat+k]==1:
            not_list.append(k)
    ci_val=0
    for p in range(k_sat):
        if p in not_list:
            x_list[p]=not x_list[p]
    for x in x_list:
        ci_val=ci_val or x
    acc.append(ci_val)

acc_np=np.array(acc)
print(acc_np)
print(np.mean(acc_np))

'''
