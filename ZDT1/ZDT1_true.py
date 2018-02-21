# True Pareto frontier of ZDT1
# f1 = 0:0.01:1
# f2 = 1 - sqrt(f1)
from __future__ import division, print_function, absolute_import
import numpy as np

def pareto():
    n = 100
    f = np.zeros([n,2])
    f[:,0] = np.linspace(0,1,n)
    f[:,1] = 1.0 - np.sqrt(f[:,0])
    return f
