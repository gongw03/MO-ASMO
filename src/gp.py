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
            smlist.append(GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=5))
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
