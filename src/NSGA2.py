# Nondominated Sorting Genetic Algorithm II (NSGA-II)
# An multi-objective optimization algorithm
from __future__ import division, print_function, absolute_import
import sampling
import numpy as np
import copy

def optimization(model, nInput, nOutput, xlb, xub, pop, gen, \
                 crossover_rate = 0.9, mu = 20, mum = 20):
    ''' Nondominated Sorting Genetic Algorithm II, An multi-objective algorithm
        model: the evaluated model function
        nInput: number of model input
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mu: distribution index for crossover
        mum: distribution index for mutation
    '''
    poolsize = int(round(pop/2.)); # size of mating pool;
    toursize = 2;                  # tournament size;

    x = sampling.lh(pop, nInput)
    x = x * (xub - xlb) + xlb
    y = np.zeros((pop, nOutput))
    for i in range(pop):
        y[i,:] = model.evaluate(x[i,:])
    icall = pop

    x, y, rank, crowd = sortMO(x, y, nInput, nOutput)
    population_para = x.copy()
    population_obj  = y.copy()

    for i in range(gen):
        pool = selection(population_para, population_obj, nInput, pop, poolsize, toursize)
        count = 0
        while (count < pop - 1):
            if (np.random.rand() < 0.9):
                parentidx = np.random.choice(poolsize, 2, replace = False)
                parent1   = pool[parentidx[0],:]
                parent2   = pool[parentidx[1],:]
                child1, child2 = crossover(parent1, parent2, mu, xlb, xub)
                y1 = model.evaluate(child1)
                y2 = model.evaluate(child2)
                x  = np.vstack((x,child1,child2))
                y  = np.vstack((y,y1,y2))
                population_para = np.vstack((population_para,child1,child2))
                population_obj  = np.vstack((population_obj,y1,y2))
                count += 2
                icall += 2
            else:
                parentidx = np.random.randint(poolsize)
                parent    = pool[parentidx,:]
                child     = mutation(parent, mum, xlb, xub)
                y1 = model.evaluate(child)
                x  = np.vstack((x,child))
                y  = np.vstack((y,y1))
                population_para = np.vstack((population_para,child))
                population_obj  = np.vstack((population_obj,y1))
                count += 1
                icall += 1
        population_para, population_obj = \
            remove_worst(population_para, population_obj, pop, nInput, nOutput)
        bestx = population_para.copy()
        besty = population_obj.copy()
    return bestx, besty, x, y

def sortMO(x, y, nInput, nOutput):
    ''' Non domination sorting for multi-objective optimization
        x: input parameter matrix
        y: output objectives matrix
        nInput: number of input
        nOutput: number of output
    '''
    rank, dom = fast_non_dominated_sort(y)
    idxr = rank.argsort()
    rank = rank[idxr]
    x = x[idxr,:]
    y = y[idxr,:]
    T = x.shape[0]

    crowd = np.zeros(T)
    rmax = int(rank.max())
    idxt = np.zeros(T, dtype = np.int)
    c = 0
    for k in range(rmax):
        rankidx = (rank == k)
        D = crowding_distance(y[rankidx,:])
        idxd = D.argsort()[::-1]
        crowd[rankidx] = D[idxd]
        idxtt = np.array(range(len(rank)))[rankidx]
        idxt[c:(c+len(idxtt))] = idxtt[idxd]
        c += len(idxtt)
    x = x[idxt,:]
    y = y[idxt,:]
    rank = rank[idxt]

    return x, y, rank, crowd

def fast_non_dominated_sort(Y):
    ''' a fast non-dominated sorting method
        Y: output objective matrix
    '''
    N, d = Y.shape
    Q = [] # temp array of Pareto front index
    Sp = [] # temp array of points dominated by p
    S = [] # temp array of Sp
    rank = np.zeros(N) # Pareto rank
    n = np.zeros(N)  # domination counter of p
    dom = np.zeros((N, N))  # the dominate matrix, 1: i doms j, 2: j doms i

    # compute the dominate relationship online, much faster
    for i in range(N):
        for j in range(N):
            if i != j:
                if dominates(Y[i,:], Y[j,:]):
                    dom[i,j] = 1
                    Sp.append(j)
                elif dominates(Y[j,:], Y[i,:]):
                    dom[i,j] = 2
                    n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            Q.append(i)
        S.append(copy.deepcopy(Sp))
        Sp = []

    F = []
    F.append(copy.deepcopy(Q))
    k = 0
    while len(F[k]) > 0:
        Q = []
        for i in range(len(F[k])):
            p = F[k][i]
            for j in range(len(S[p])):
                q = S[p][j]
                n[q] -= 1
                if n[q] == 0:
                    rank[q]  = k + 1
                    Q.append(q)
        k += 1
        F.append(copy.deepcopy(Q))

    return rank, dom

def dominates(p,q):
    ''' comparison for multi-objective optimization
        d = True, if p dominates q
        d = False, if p not dominates q
        p and q are 1*nOutput array
    '''
    if sum(p > q) == 0:
        d = True
    else:
        d = False
    return d

def crowding_distance(Y):
    ''' compute crowding distance in NSGA-II
        Y is the output data matrix
        [n,d] = size(Y)
        n: number of points
        d: number of dimentions
    '''
    n,d = Y.shape
    lb = np.min(Y, axis = 0)
    ub = np.max(Y, axis = 0)

    if n == 1 or np.min(ub-lb) == 0.0:
        D = np.array([1.])
    else:
        U = (Y - lb) / (ub - lb)

        D = np.zeros(n)
        DS = np.zeros((n,d))

        idx = U.argsort(axis = 0)
        US = np.zeros((n,d))
        for i in range(d):
            US[:,i] = U[idx[:,i],i]

        DS[0,:] = 1.
        DS[n-1,:] = 1.

        for i in range(1,n-1):
            for j in range(d):
                DS[i,j] = US[i+1,j] - US[i-1,j]

        for i in range(n):
            for j in range(d):
                D[idx[i,j]] += DS[i,j]
        D[np.isnan(D)] = 0.0

    return D

def mutation(parent, mum, xlb, xub):
    ''' Polynomial Mutation in Genetic Algorithm
        For more information about PMut refer the NSGA-II paper.
        mum: distribution index for mutation, default = 20
            This determine how well spread the child will be from its parent.
        parent: sample point before mutation
	'''
    n     = len(parent)
    delta = np.ndarray(n)
    child = np.ndarray(n)
    u     = np.random.rand(n)
    for i in range(n):
        if (u[i] < 0.5):
            delta[i] = (2.0*u[i])**(1.0/(mum+1)) - 1.0
        else:
            delta[i] = 1.0 - (2.0*(1.0 - u[i]))**(1.0/(mum+1))
        child[i] = parent[i] + (xub[i] - xlb[i]) * delta[i]
    child = np.clip(child, xlb, xub)
    return child

def crossover(parent1, parent2, mu, xlb, xub):
    ''' SBX (Simulated Binary Crossover) in Genetic Algorithm
        For more information about SBX refer the NSGA-II paper.
        mu: distribution index for crossover, default = 20
        This determine how well spread the children will be from their parents.
    '''
    n      = len(parent1)
    beta   = np.ndarray(n)
    child1 = np.ndarray(n)
    child2 = np.ndarray(n)
    u = np.random.rand(n)
    for i in range(n):
        if (u[i] <= 0.5):
            beta[i] = (2.0*u[i])**(1.0/(mu+1))
        else:
            beta[i] = (1.0/(2.0*(1.0 - u[i])))**(1.0/(mu+1))
        child1[i] = 0.5*((1-beta[i])*parent1[i] + (1+beta[i])*parent2[i])
        child2[i] = 0.5*((1+beta[i])*parent1[i] + (1-beta[i])*parent2[i])
    child1 = np.clip(child1, xlb, xub)
    child2 = np.clip(child2, xlb, xub)
    return child1, child2

def selection(population_para, population_obj, nInput, pop, poolsize, toursize):
    ''' tournament selecting the best individuals into the mating pool'''
    pool    = np.zeros([poolsize,nInput])
    poolidx = np.zeros(poolsize)
    count   = 0
    while (count < poolsize-1):
        candidate = np.random.choice(pop, toursize, replace = False)
        idx = candidate.min()
        if not(idx in poolidx):
            poolidx[count] = idx
            pool[count,:]  = population_para[idx,:]
            count += 1
    return pool

def remove_worst(population_para, population_obj, pop, nInput, nOutput):
    ''' remove the worst individuals in the population '''
    population_para, population_obj, rank, crowd = \
        sortMO(population_para, population_obj, nInput, nOutput)
    return population_para[0:pop,:], population_obj[0:pop,:]

