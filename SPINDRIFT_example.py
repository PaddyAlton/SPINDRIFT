# SPINDRIFT_example.py
# script for demonstrating SPINDRIFT_MCMC functionality
# PaddyAlton -- 2017-10-06

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import astropy.io.fits as fits
import scipy.optimize as opt
import imp

import emcee
import corner
import seaborn as sns; sns.set_context("paper"); sns.set_style("white"); sns.set_style("ticks")

import SPINDRIFT_MCMC as mcmc
import SPINDRIFT_index_model as bpl

# Here is a list of parameter labels for the full set of parameters available to SPINDRIFT_MCMC:
par_labels = np.array(["log(age)", "X1", " X2", "[Z/H]", "[Mg/H]", "[Ca/H]", "[Na/H]", "[K/H]", "[C/H]", "[Fe/H]", "[Ti/H]", "$\Delta$T_eff", "[ONeS/H]"])

# Here are some definitions for the low and high edges of the prior box imposed on the parameters:
# (note that the SPINDRIFT_MCMC code can easily be modified to make a more sophisticated prior function, but the default version just uses a 'top-hat' approach)
#bpl.predictor inputs: l_age,      X1, X2,    Z, p_mg, p_ca, p_na, p_k, p_cbn, p_fe, p_ti, T_eff, p_ONeS
p_min = np.array([ np.log(7),    1.0, 1.0, -1.0, -0.4, -0.4, -0.6, -0.3, -0.3, -0.3, -0.4, -100, -0.4 ])
p_max = np.array([ np.log(14.0), 3.5, 3.5, +0.5, +0.6, +0.4, +1.2, +0.3, +0.3, +0.3, +0.6, +100, +0.4 ])

p_fid = np.array([np.log(13.5),  1.3, 2.3, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  0.00]) #fiducial parameter values

fid_predict = bpl.predictor(np.ones(len(p_fid),bool),p_fid) # predict some line strengths corresponding to the fiducial parameter values

# Here are the labels for the spectroscopic features I include in my models:
feat_labels = np.array(['C$_2$','H$\\beta$','Fe5015', 'Mgb', 'FeI 0.52', 'FeI 0.53','NaI 0.59', 'NaI 0.82','CaII Triplet', 'MgI 0.88', 'FeH', 'CaI 1.06', 
                        'NaI 1.14', 'KI 1.17', 'KI 1.25', 'AlI 1.31', 'CaI 1.98', 'NaI 2.21', 'CaI 2.26', 'CO bandhead'])

par_mask = np.ones(len(p_min),bool) # the parameter mask has a boolean value associated with each parameter. If False, that parameter is switched off and held at the fiducial value.

par_mask[11] = False  # T_eff is irrelevant when you fit Z, as this parameter becomes completely degenerate (you'd need it on if you were to fit at fixed [Z/H]=0)

mask = np.ones(len(fid_predict),bool) # the feature mask has a boolean value associated with each feature. If False, that feature is ignored in the fit (e.g. if it's contaminated)

mask[0] = False # no_data
mask[4] = False # no_data
mask[5] = False # no_data
mask[6] = False # no_data
mask[7] = False # no_data
mask[15]= False # don't use AlI (no [Al/Fe] parameter available)
mask[9]  = False
mask[-2] = False
mask[-1] = False

### SET OUT THE PARAMETERS OF THE MCMC RUN:
n_walkers=100
n_steps = 1250 
n_burn  = 250

#### I guess we should create some fake data...

uncertainties = fid_predict*0.05 # 5% errors on each index
data = fid_predict + np.random.randn(len(fid_predict))*uncertainties

#### call MCMC routine

instance = mcmc.emcee_wrapper(data,uncertainties, mask, par_mask, p_min, p_max, 'SPINDRIFT_test')

instance.MLEstRoutine()
instance.Runner(n_walkers,n_steps,n_burn)
instance.ComputeResults()
instance.OutputTable()