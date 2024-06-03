import math
import mRobot
import mGP
import mComputing

import torch
import gpytorch
import pandas as pd
# import matplotlib as mpl
# # mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import time


N  = 10; T = 1.0; H = 3; L = 40; MAX_ITER = 100
x_min =  0.; x_max = 200.
y_min =  0.; y_max = 200.
s_max = 10.; R = 40.; r = 3.


pBounds = mRobot.polyBound(s_max, x_min, x_max, y_min, y_max)
init = mRobot.init_position(pBounds, R, r, N)
robo = [mRobot.robot(i, T, H, R, r, 0., pBounds, init[i][:]) for i in  range(N)]

dataframe  = pd.read_csv('SOM.csv', header=None)
data       = torch.tensor(dataframe.values)
train_posn = data[:,[0,1]]
train_meas = data[:,2]


GTlikelihood = gpytorch.likelihoods.GaussianLikelihood(num_tasks=1)
GTmodel  = mGP.ExactGPModel(train_posn, train_meas, GTlikelihood)


GTmodel.train()
GTlikelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(GTlikelihood, GTmodel)
mGP.mtrain(GTmodel, mll)



for i in range(N):
    robo[i].cmea = mGP.measure(GTmodel, GTlikelihood, torch.tensor([robo[i].cpos]))

arrLlh = []
arrMod = []
for i in range(N):
    neighbors  = mRobot.find_nears(robo)
    train_meas = robo[i].cmea
    train_posn = torch.tensor([robo[i].cpos], dtype=torch.double)
    for j in neighbors[i]:
        train_meas = torch.cat((train_meas, robo[j].cmea), 0)
        train_posn = torch.cat((train_posn, torch.tensor([robo[j].cpos], dtype=torch.double)), 0)
    likelihood   = gpytorch.likelihoods.GaussianLikelihood(num_tasks=1)
    model        = mGP.ExactGPModel(train_posn, train_meas, likelihood)
    arrLlh.append(likelihood)
    arrMod.append(model)   
   
    
for i in range(N):
    arrLlh[i].eval()
    arrMod[i].eval()
    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
        point =  torch.tensor([[100,100]])
        predictions = arrLlh[i](arrMod[i](point))


for i in range(2):
    mComputing.rand_position(robo)

    for i in range(N):    
        robo[i].cmea = mGP.measure(GTmodel, GTlikelihood, torch.tensor([robo[i].cpos]))
        
    mComputing.update(robo, arrMod)
    for i in range(N):
        arrMod[i].train()
        arrLlh[i].train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(arrLlh[i], arrMod[i])
        mGP.mtrain(arrMod[i], mll)


GTmodel.eval()
GTlikelihood.eval()
n1, n2 = 100, 100
xv, yv = torch.meshgrid(torch.linspace(0, 400, n1), torch.linspace(0, 400, n2), indexing="ij")

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
    test_x = torch.stack([xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
    predictions = GTlikelihood(GTmodel(test_x))
    mean = predictions.mean
    
extent = (xv.min(), xv.max(), yv.min(), yv.max())    

plt.clf()
plt.imshow(mean.detach().numpy().reshape(n1, n2), extent=extent)
plt.savefig('testplot.png')

plt.show()
