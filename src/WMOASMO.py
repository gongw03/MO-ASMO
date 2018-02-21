# Weighted Multi-Objective Adaptive Surrogate Modelling-based Optimization
from __future__ import division, print_function, absolute_import
import sampling
import gp
import WNSGA2
import numpy as np

def optimization(model, nInput, nOutput, xlb, xub, dft, niter, pct, \
                 Xinit = None, Yinit = None, pop = 100, gen = 100, \
                 crossover_rate = 0.9, mu = 20, mum = 20, weight = 0.001):
    ''' Weighted Multi-Objective Adaptive Surrogate Modelling-based Optimization
        model: the evaluated model function
        nInput: number of model input
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        dft: default point to constrained the objective space, 
             objective value simulated by default parameters
             (dimension equals to nOutput, this is the reference point in MO-ASMO paper)
        niter: number of iteration
        pct: percentage of resampled points in each iteration
        Xinit and Yinit: initial samplers for surrogate model construction
        ### options for the embedded WNSGA-II of WMO-ASMO
            pop: number of population
            gen: number of generation
            crossover_rate: ratio of crossover in each generation
            mu: distribution index for crossover
            mum: distribution index for mutation
            weight: assign weight factor if one objective is worse than the dft point
    '''
    N_resample = int(pop*pct)
    if (Xinit is None and Yinit is None):
        Ninit = nInput * 10
        Xinit = sampling.glp(Ninit, nInput)
        for i in range(Ninit):
            Xinit[i,:] = Xinit[i,:] * (xub - xlb) + xlb
        Yinit = np.zeros((Ninit, nOutput))
        for i in range(Ninit):
            Yinit[i,:] = model.evaluate(Xinit[i,:])
    else:
        Ninit = Xinit.shape[0]
    icall = Ninit
    x = Xinit.copy()
    y = Yinit.copy()

    for i in range(niter):
        print('Surrogate Opt loop: %d' % i)
        sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub)
        bestx_sm, besty_sm, x_sm, y_sm = \
            WNSGA2.optimization(sm, nInput, nOutput, xlb, xub, dft, \
                                pop, gen, crossover_rate, mu, mum, weight)
        D = WNSGA2.weighted_crowding_distance(besty_sm, dft, weight)
        idxr = D.argsort()[::-1][:N_resample]
        x_resample = bestx_sm[idxr,:]
        y_resample = np.zeros((N_resample,nOutput))
        for j in range(N_resample):
            y_resample[j,:] = model.evaluate(x_resample[j,:])
        icall += N_resample
        x = np.vstack((x, x_resample))
        y = np.vstack((y, y_resample))

    xtmp = x.copy()
    ytmp = y.copy()
    xtmp, ytmp, rank, crowd = WNSGA2.sortMO_W(xtmp, ytmp, nInput, nOutput, dft, weight)
    bestx = []
    besty = []
    for i in range(ytmp.shape[0]):
        if rank[i] == 0 and sum(ytmp[i,:] < dft) == nOutput:
            bestx.append(xtmp[i,:])
            besty.append(ytmp[i,:])
    bestx = np.array(bestx)
    besty = np.array(besty)
    #idxp = (rank == 0)
    #bestx = xtmp[idxp,:]
    #besty = ytmp[idxp,:]

    return bestx, besty, x, y

def onestep(nInput, nOutput, xlb, xub, dft, pct, \
            Xinit, Yinit, pop = 100, gen = 100, \
            crossover_rate = 0.9, mu = 20, mum = 20, weight = 0.001):
    ''' Weighted Multi-Objective Adaptive Surrogate Modelling-based Optimization
        One-step mode for offline optimization
        Do NOT call the model evaluation function
        nInput: number of model input
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        dft: default point to constrained the objective space, 
             objective value simulated by default parameters
             (dimension equals to nOutput, this is the reference point in MO-ASMO paper)
        pct: percentage of resampled points in each iteration
        Xinit and Yinit: initial samplers for surrogate model construction
        ### options for the embedded WNSGA-II of WMO-ASMO
            pop: number of population
            gen: number of generation
            crossover_rate: ratio of crossover in each generation
            mu: distribution index for crossover
            mum: distribution index for mutation
            weight: assign weight factor if one objective is worse than the dft point
    '''
    N_resample = int(pop*pct)
    Ninit = Xinit.shape[0]
    x = Xinit.copy()
    y = Yinit.copy()
    sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub)
    bestx_sm, besty_sm, x_sm, y_sm = \
        WNSGA2.optimization(sm, nInput, nOutput, xlb, xub, dft, \
                            pop, gen, crossover_rate, mu, mum, weight)
    D = WNSGA2.weighted_crowding_distance(besty_sm, dft, weight)
    idxr = D.argsort()[::-1][:N_resample]
    x_resample = bestx_sm[idxr,:]
    return x_resample

