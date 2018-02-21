from __future__ import division, print_function, absolute_import
import os
import sys
sys.path.append('../src')
import MOASMO
import numpy as np
import matplotlib.pyplot as plt
import util
import cPickle

# model name 
modelname = 'ZDT1'
model = __import__(modelname)

# result path
respath = '../../UQ-res/MOO/%s' % modelname
if not os.path.exists(respath):
    os.makedirs(respath)

# load parameter name and range
pf = util.read_param_file('%s.txt' % modelname)
bd = np.array(pf['bounds'])
nInput = pf['num_vars']
nOutput = 2
xlb = bd[:,0]
xub = bd[:,1]

# parameters for MO-ASMO
niter = 5
pct = 0.2

# run MO-ASMO
bestx, besty, x, y = \
    MOASMO.optimization(model, nInput, nOutput, xlb, xub, niter, pct)

# plot results
plt.plot(y[:,0],y[:,1],'b.',label='evaluated points')
plt.plot(besty[:,0],besty[:,1],'r.',label='MO-ASMO optimal')

model_true = __import__(modelname + '_true')
y_true = model_true.pareto()
plt.plot(y_true[:,0],y_true[:,1],'k-',label='True Pareto')

plt.xlabel('y1')
plt.ylabel('y2')
plt.legend()

# save figure
plt.savefig('%s/ZDT1_MOASMO.png' % respath)

# save results to bin file
with open('%s/ZDT1_MOASMO.bin' % respath, 'w') as f:
    cPickle.dump({'bestx': bestx, 'besty': besty, 'x': x, 'y': y}, f)
