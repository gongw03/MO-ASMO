# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:18:23 2014
@author: gongwei
"""
# Good Lattice Points method for Uniform Design
from __future__ import division, print_function, absolute_import
import numpy as np
import fractions as fc
import itertools
from discrepancy import CD2
#from discrepancy_cython import CD2

def sample(n, s):
    ''' main function of GLP design'''
    m = EulerFunction(n)
    if float(m)/n < 0.9:
        if m < 20 and s < 4:
            m = EulerFunction(n+1)
            X = GLP_GV(n+1,s,m,plusone=True)
        else:
            X = GLP_PGV(n+1,s,plusone=True)
    else:
        if m < 20 and s < 4:
            X = GLP_GV(n,s,m)
        else:
            X = GLP_PGV(n,s)
    return X

def GLP_PGV(n,s,plusone=False):
    ''' type 2 GLP design, if the combination of C(s,m) is large'''
    h = PowerGenVector(n,s)
    X = np.random.uniform(0,1,size=[n,s])
    D = 1e32
    #for i in range(min(h.shape[0],20)):
    for i in range(h.shape[0]):
        x = glpmod(n,h[i,:])
        if plusone:
            x = x[0:n-1,:]
            x = (x - 0.5)/(n-1)
        else:
            x = (x - 0.5)/n
        d = CD2(x)
        if d < D:
            D = d
            X = x
    return X

def GLP_GV(n,s,m,plusone=False):
    ''' type 1 GLP design, if the combination of C(s,m) is small'''
    h = GenVector(n)
    u = glpmod(n,h)
    clist = itertools.combinations(range(m),s)
    X = np.random.uniform(0,1,size=[n,s])
    D = 1e32
    for c in clist:
        if plusone:
            x = u[0:n-1,c]
            x = (x - 0.5)/(n-1)
        else:
            x = u[:,c]
            x = (x - 0.5)/n
        d = CD2(x)
        if d < D:
            D = d
            X = x
    return X

def PrimeFactors(n):
    '''generate all prime factors of n'''
    p = []
    f = 2
    while f < n:
        while not n%f:
            p.append(f)
            n //= f
        f += 1
    if n > 1:
        p.append(n)

    return p

def EulerFunction(n):
    p = PrimeFactors(n)
    fai = n*(1-1.0/p[0])
    for i in range(1,len(p)):
        if p[i] != p[i-1]:
            fai *= 1-1.0/p[i]
    return int(fai)

def GenVector(n):
    h = []
    for i in range(n):
        if fc.gcd(i,n) == 1:
            h.append(i)
    return h

def PowerGenVector(n,s):
    a = []
    for i in range(2,n):
        if fc.gcd(i,n) == 1:
            a.append(i)
    aa = []
    #for i in range(min(len(a),20)):
    for i in range(len(a)):
        ha = np.mod([a[i]**t for t in range(1,s)],n)
        ha = np.sort(ha)
        rep = False
        if ha[0] == 1:
            rep = True
        for j in range(1,len(ha)):
            if ha[j] == ha[j-1]:
                rep = True
        if rep == False:
            aa.append(a[i])

    hh = np.zeros([len(aa),s])
    for i in range(len(aa)):
        hh[i,:] = np.mod([aa[i]**t for t in range(s)],n)
    return hh

def glpmod(n,h):
    ''' generate GLP using generation vector h'''
    m = len(h)
    u = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            u[i,j] = np.mod((i+1)*h[j],n)
            if u[i,j] == 0:
                u[i,j] = n
    return u
