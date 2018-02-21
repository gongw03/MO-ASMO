from __future__ import division, print_function, absolute_import
import numpy as np
def evaluate(x):
    ''' This is the Zitzler-Deb-Thiele Function - type A
        Bound: XUB = [1,1,...]; XLB = [0,0,...]
        dim = 30
    '''
    f = np.zeros(2)
    f[0] = x[0]
    g = 1. + 9./29.*np.sum(x[1:])
    h = 1. - np.sqrt(f[0]/g)
    f[1] = g*h
    return f

