from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
#from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel)
import copy


class GPR_Matern:
    def __init__(self, xin, yin, nInput, nOutput, N, xlb, xub):
        self.nInput  = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrg = xub - xlb

        x = copy.deepcopy(xin)
        y = copy.deepcopy(yin)
        for i in range(N):
            x[i,:] = (x[i,:] - self.xlb) / self.xrg
        if nOutput == 1:
            y = y.reshape((y.shape[0],1))

        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)

        smlist = []
        for i in range(nOutput):
            #smlist.append(GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=5))
            smlist.append(GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer=sceua_optimizer))
            smlist[i].fit(x,y[:,i])
        self.smlist = smlist

    def predict(self,xin):
        x = copy.deepcopy(xin)
        if len(x.shape) == 1: x = x.reshape((1,self.nInput))
        N = x.shape[0]
        y = np.zeros((N,self.nOutput))
        for i in range(N):
            x[i,:] = (x[i,:] - self.xlb) / self.xrg
        for i in range(self.nOutput):
            y[:,i] = self.smlist[i].predict(x)
        return y

    def evaluate(self,x):
        return self.predict(x)

def sceua_optimizer(obj_func, initial_theta, bounds):
    """
    SCE-UA optimizer for optimizing hyper parameters of GPR
    Input:
      * 'obj_func' is the objective function to be maximized, which
        takes the hyperparameters theta as parameter and an
        optional flag eval_gradient, which determines if the
        gradient is returned additionally to the function value
      * 'initial_theta': the initial value for theta, which can be 
        used by local optimizers
      * 'bounds': the bounds on the values of theta, 
        [(lb1, ub1), (lb2, ub2), (lb3, ub3)]
     Returned:
      * 'theta_opt' is the best found hyperparameters theta
      * 'func_min' is the corresponding value of the target function.
    """
    nopt = len(bounds)
    bl = np.zeros(nopt)
    bu = np.zeros(nopt)
    for i,bd in enumerate(bounds):
        bl[i] = bd[0]
        bu[i] = bd[1]
    ngs = nopt
    maxn = 3000
    kstop = 10
    pcento = 0.1
    peps = 0.001
    verbose = False
    [bestx, bestf, icall, nloop, bestx_list, bestf_list, icall_list] = \
        sceua(obj_func, bl, bu, nopt, ngs, maxn, kstop, pcento, peps, verbose)
    theta_opt = bestx
    func_min = bestf
    return theta_opt, func_min


def sceua(func, bl, bu, nopt, ngs, maxn, kstop, pcento, peps, verbose):
    """
    This is the subroutine implementing the SCE algorithm, 
    written by Q.Duan, 9/2004
    translated to python by gongwei, 11/2017

    Parameters:
    func:   optimized function
    bl:     the lower bound of the parameters
    bu:     the upper bound of the parameters
    nopt:   number of adjustable parameters
    ngs:    number of complexes (sub-populations)
    maxn:   maximum number of function evaluations allowed during optimization
    kstop:  maximum number of evolution loops before convergency
    pcento: the percentage change allowed in kstop loops before convergency
    peps:   relative size of parameter space
    
    npg:  number of members in a complex 
    nps:  number of members in a simplex
    npt:  total number of points in an iteration
    nspl:  number of evolution steps for each complex before shuffling
    mings: minimum number of complexes required during the optimization process

    LIST OF LOCAL VARIABLES
    x[.,.]:    coordinates of points in the population
    xf[.]:     function values of x[.,.]
    xx[.]:     coordinates of a single point in x
    cx[.,.]:   coordinates of points in a complex
    cf[.]:     function values of cx[.,.]
    s[.,.]:    coordinates of points in the current simplex
    sf[.]:     function values of s[.,.]
    bestx[.]:  best point at current shuffling loop
    bestf:     function value of bestx[.]
    worstx[.]: worst point at current shuffling loop
    worstf:    function value of worstx[.]
    xnstd[.]:  standard deviation of parameters in the population
    gnrng:     normalized geometri%mean of parameter ranges
    lcs[.]:    indices locating position of s[.,.] in x[.,.]
    bound[.]:  bound on ith variable being optimized
    ngs1:      number of complexes in current population
    ngs2:      number of complexes in last population
    criter[.]: vector containing the best criterion values of the last 10 shuffling loops
    """

    # Initialize SCE parameters:
    npg  = 2 * nopt + 1
    nps  = nopt + 1
    nspl = npg
    npt  = npg * ngs
    bd   = bu - bl

    # Create an initial population to fill array x[npt,nopt]
    x = np.random.random([npt,nopt])
    for i in range(npt):
        x[i,:] = x[i,:] * bd + bl

    xf = np.zeros(npt)
    for i in range(npt):
        xf[i] = func(x[i,:])[0] # only used the first returned value
    icall = npt

    # Sort the population in order of increasing function values
    idx = np.argsort(xf)
    xf = xf[idx]
    x = x[idx,:]

    # Record the best and worst points
    bestx  = copy.deepcopy(x[0,:])
    bestf  = copy.deepcopy(xf[0])
    worstx = copy.deepcopy(x[-1,:])
    worstf = copy.deepcopy(xf[-1])
    
    bestf_list = []
    bestf_list.append(bestf)
    bestx_list = []
    bestx_list.append(bestx)
    icall_list = []
    icall_list.append(icall)
    
    if verbose:
        print('The Initial Loop: 0')
        print('BESTF  : %f' % bestf)
        print('BESTX  : %s' % np.array2string(bestx))
        print('WORSTF : %f' % worstf)
        print('WORSTX : %s' % np.array2string(worstx))
        print(' ')

    # Computes the normalized geometric range of the parameters
    gnrng = np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bd)))
    # Check for convergency
    if verbose:
        if icall >= maxn:
            print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
            print('ON THE MAXIMUM NUMBER OF TRIALS ')
            print(maxn)
            print('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
            print(icall)
            print('OF THE INITIAL LOOP!')

        if gnrng < peps:
            print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')
    
    # Begin evolution loops:
    nloop = 0
    criter = []
    criter_change = 1e+5
    cx = np.zeros([npg,nopt])
    cf = np.zeros(npg)

    while (icall < maxn) and (gnrng > peps) and (criter_change > pcento):
        nloop += 1
        
        # Loop on complexes (sub-populations)
        for igs in range(ngs):
        
            # Partition the population into complexes (sub-populations)
            k1 = np.int64(np.linspace(0, npg-1, npg))
            k2 = k1 * ngs + igs
            cx[k1,:] = copy.deepcopy(x[k2,:])
            cf[k1] = copy.deepcopy(xf[k2])
            
            # Evolve sub-population igs for nspl steps
            for loop in range(nspl):
                
                # Select simplex by sampling the complex according to a linear
                # probability distribution
                lcs = np.zeros(nps, dtype=np.int64)
                lcs[0] = 0
                for k3 in range(1,nps):
                    for itmp in range(1000):
                        lpos = int(np.floor(
                                npg + 0.5 - np.sqrt((npg + 0.5)**2 - 
                                npg * (npg + 1) * np.random.rand())))
                        if len(np.where(lcs[:k3] == lpos)[0]) == 0:
                            break
                    lcs[k3] = lpos
                lcs = np.sort(lcs)
    
                # Construct the simplex:
                s = copy.deepcopy(cx[lcs,:])
                sf = copy.deepcopy(cf[lcs])
                
                snew, fnew, icall = cceua(func, s, sf, bl, bu, icall)
    
                # Replace the worst point in Simplex with the new point:
                s[nps-1,:] = snew
                sf[nps-1] = fnew
                
                # Replace the simplex into the complex
                cx[lcs,:] = copy.deepcopy(s)
                cf[lcs] = copy.deepcopy(sf)
                
                # Sort the complex
                idx = np.argsort(cf)
                cf = cf[idx]
                cx = cx[idx,:]
                
            # End of Inner Loop for Competitive Evolution of Simplexes
    
            # Replace the complex back into the population
            x[k2,:] = copy.deepcopy(cx[k1,:])
            xf[k2] = copy.deepcopy(cf[k1])
        
        # End of Loop on Complex Evolution;
        
        # Shuffled the complexes
        idx = np.argsort(xf)
        xf = xf[idx]
        x = x[idx,:]
        
        # Record the best and worst points
        bestx  = copy.deepcopy(x[0,:])
        bestf  = copy.deepcopy(xf[0])
        worstx = copy.deepcopy(x[-1,:])
        worstf = copy.deepcopy(xf[-1])
        bestf_list.append(bestf)
        bestx_list.append(bestx)
        icall_list.append(icall)
        
        if verbose:
            print('Evolution Loop: %d - Trial - %d' % (nloop, icall))
            print('BESTF  : %f' % bestf)
            print('BESTX  : %s' % np.array2string(bestx))
            print('WORSTF : %f' % worstf)
            print('WORSTX : %s' % np.array2string(worstx))
            print(' ')
        
        # Computes the normalized geometric range of the parameters
        gnrng = np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bd)))
        # Check for convergency;
        if verbose:
            if icall >= maxn:
                print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                print('ON THE MAXIMUM NUMBER OF TRIALS %d HAS BEEN EXCEEDED!' % maxn)
            if gnrng < peps:
                print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')
            
    
        criter.append(bestf)
        if nloop >= kstop:
            criter_change = np.abs(criter[nloop-1] - criter[nloop-kstop])*100
            criter_change /= np.mean(np.abs(criter[nloop-kstop:nloop]))
            if criter_change < pcento:
                if verbose:
                    print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY LESS THAN THE THRESHOLD %f%%' % (kstop, pcento))
                    print('CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')
        
    # End of the Outer Loops
    
    if verbose:
        print('SEARCH WAS STOPPED AT TRIAL NUMBER: %d' % icall )
        print('NORMALIZED GEOMETRIC RANGE = %f' % gnrng )
        print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY %f%%' % (kstop, criter_change))
   
    # END of Subroutine SCEUA_runner
    return bestx, bestf, icall, nloop, bestx_list, bestf_list, icall_list


def cceua(func, s, sf, bl, bu, icall):
    """
    This is the subroutine for generating a new point in a simplex
    func:   optimized function
    s[.,.]: the sorted simplex in order of increasing function values
    sf[.]:  function values in increasing order

    LIST OF LOCAL VARIABLES
    sb[.]:   the best point of the simplex
    sw[.]:   the worst point of the simplex
    w2[.]:   the second worst point of the simplex
    fw:      function value of the worst point
    ce[.]:   the centroid of the simplex excluding wo
    snew[.]: new point generated from the simplex
    iviol:   flag indicating if constraints are violated
             = 1 , yes
             = 0 , no
    """

    nps, nopt = s.shape
    n = nps
    alpha = 1.0
    beta = 0.5

    # Assign the best and worst points:
    sw = s[-1,:]
    fw = sf[-1]

    # Compute the centroid of the simplex excluding the worst point:
    ce = np.mean(s[:n-1,:],axis=0)

    # Attempt a reflection point
    snew = ce + alpha * (ce - sw)

    # Check if is outside the bounds:
    ibound = 0
    s1 = snew - bl
    if sum(s1 < 0) > 0:
        ibound = 1
    s1 = bu - snew
    if sum(s1 < 0) > 0:
        ibound = 2
    if ibound >= 1:
        snew = bl + np.random.random(nopt) * (bu - bl)

    fnew = func(snew)[0] # only used the first returned value
    icall += 1

    # Reflection failed; now attempt a contraction point
    if fnew > fw:
        snew = sw + beta * (ce - sw)
        fnew = func(snew)[0] # only used the first returned value
        icall += 1
    
    # Both reflection and contraction have failed, attempt a random point
        if fnew > fw:
            snew = bl + np.random.random(nopt) * (bu - bl)
            fnew = func(snew)[0] # only used the first returned value
            icall += 1

    # END OF CCE
    return snew, fnew, icall
