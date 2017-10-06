# SPINDRIFT_mcmc.py
# callable functions for MCMC using the up-to-date, two-part power law CvD-16 models.
# PaddyAlton -- 2017-03-06 (v1.0)

import numpy as np
import astropy.io.fits as fits
import glob as glob
import os
import imp
import subprocess as subp
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white"); sns.set_color_codes(palette='colorblind')
import scipy.optimize as opt
import emcee
import corner

from scipy.interpolate import RegularGridInterpolator as rg_int
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline as rbsplin

import SPINDRIFT_index_model as bpl

from scipy.stats import gaussian_kde as kde
from numpy.lib.recfunctions import append_fields

### ESTABLISH OUTPUT DIRECTORIES ###

if os.path.isdir('emcee_plots/') & os.path.isdir('results/'): 
    plot_dir = 'emcee_plots/'
    res_dir = 'results/'
else:
    print 'Directories for storing plots and MCMC results do not exist under their default names.'
    plot_dir = raw_input('Please enter a path to a directory to store plots in')
    res_dir = raw_input('Please enter a path to a directory to store MCMC results in')
    if not plot_dir.endswith('/'): plot_dir += '/' # this avoids an error if the user doesn't leave a trailing '/' on 
    if not res_dir.endswith('/'): res_dir += '/'   # the paths they enter.

### SOME USEFUL FUNCTIONS:

def fd_find(X1,X2,return_grid=False):

    """
    Use this function for quick evaluations of f_dwarf @ X1, X2. It is taking 
    the pre-computed BPL f_dwarf grid (t=13.5Gyr) and interpolating it smoothly.
    It works for X1,X2 {- {0.5,3.5}.

    INPUTS: X1, X2 -- IMF slopes where X1,X2 = 2.3 is Salpeter

    N.B. these can be provided as arrays, not just single values: 
    in this case the output is also an array (of f_dwarf values).

    KEYWORD: return_grid=False (if true, return 2D grid of f_dwarf values)

    OUTPUT: f_dwarf (in percent)

    """

    fd_135 = fits.getdata('BPL_fdwarf_t13.5.fits')*100. # read in pre-computed f_dwarf values for an ancient stellar population at varying values of X1, X2.

    imfx = 0.5 + np.arange(fd_135.shape[0])/5.
    
    fd_func = rbsplin(imfx,imfx,fd_135)
    
    return fd_func(X1,X2,grid=return_grid)

def xx_match():

    """ This function returns a grid of X1 values and a grid of X2 values. Any quantity computed
    as a function of IMF variations at the set of default X1, X2 values used in the CvD grid 
    can be thereby mapped onto the X1, X2 space."""
    
    imfx = 0.5 + np.arange(16)/5.

    fd_grid = fd_find(imfx,imfx,return_grid=True)

    shape_grid = np.zeros((16,16),float)
    for ii in range(16): shape_grid[ii,:] = imfx - imfx[ii]

    x1g,x2g = np.meshgrid(imfx,imfx)
    
    fd_grid2, sh_grid2 = np.meshgrid( np.linspace(1.2,32.5),np.linspace(-3,3) )
    
    cpoints = np.vstack((fd_grid.ravel(),shape_grid.ravel())).T

    x1grid = griddata(cpoints, x1g.ravel(), (fd_grid2.T,sh_grid2.T),method='linear')
    x2grid = griddata(cpoints, x2g.ravel(), (fd_grid2.T,sh_grid2.T),method='linear')
    
    x1_grid = rg_int((np.linspace(1.2,32.5),np.linspace(-3,3)),x1grid,method='linear', bounds_error=False)
    x2_grid = rg_int((np.linspace(1.2,32.5),np.linspace(-3,3)),x2grid,method='linear', bounds_error=False)
    
    return x1_grid, x2_grid


### EXPERIMENTAL LIKELIHOOD FUNCTIONS ###

def saturate_lnl(theta,data,errors,mask,par_mask,switch):

    """
    N<20 datapoints => expect less than one >2-sigma outlier
    so, saturate the penalty at 2-sigma: i.e.
    'get it within 2 sigma if you can, but if you can't,
    don't worry about getting it close.'
    """

    SP = 2 # number of sigma to saturate at, by all means change this if you want to try something else.
    values = bpl.predictor(par_mask, theta)
        
    terms = (data[mask]-values[mask])/errors[mask]
    terms[np.abs(terms)>float(SP)] = float(SP) # saturate at SP-sigma

    lnl = -0.5*np.sum(terms**2)

    if switch != 0:
        return lnl, values
    else:
        return lnl

def ratio_lnl(theta,data,errors,mask,par_mask,switch):

    """Forget the estimated uncertainties. Replace them with the predicted values --> penalise large fractional deviations instead."""

    values = bpl.predictor(par_mask, theta)
        
    if switch != 0:
        return -0.5*(np.sum(data[mask]/values[mask] - 1.)**2. ), values

    else:
        return -0.5*(np.sum(data[mask]/values[mask] - 1.)**2. )


x1_f, x2_f = xx_match() # returns two linear interpolators that yield X1, X2 values @ (fd,X2-X1)

def up_lnl(theta,data,errors,mask,par_mask,switch):

    """We may need a UNIFORM PRIOR in {f_dwarf, X2-X1}, *not* {X1,X2}"""

    XX1 = float(x1_f((theta[1],theta[2])))
    XX2 = float(x2_f((theta[1],theta[2])))

    if np.isnan(XX1): return -np.inf, 0. # use this as a 'hidden' prior; try to avoid by setting sensible prior bounds on f_dwarf and (X2-X1)

    else:

        theta2 = np.hstack((theta[0],XX1,XX2,theta[3:])) # obviously only works if age is fit for...

        values = bpl.predictor(par_mask, theta2)
        
        if switch != 0:
            return -0.5*(np.sum( ((data[mask]-values[mask])/errors[mask])**2. )), values

        else:
            return -0.5*(np.sum( ((data[mask]-values[mask])/errors[mask])**2. ))


### CORE FUNCTIONS ###

def lnprior(theta, prl, prh):

    """
    CORE FUNCTION: lnprior
    INPUTS: theta, prl, prh
    
    theta -- unpacks as a tuple of parameters (i.e. a location in parameter space to evaluate the prior at). 
    prl, prh -- label the low and high boundaries of the prior box. If any of theta lie outside the boundaries
    then ln(p) = -np.inf [p=0] is returned. Else ln(p) = 0 [p=1] is returned.
    
    Note that this is a simple 'top-hat' prior which makes things speedy. In principle a more sophisticated 
    lnprior function can be used in its place.
    """

    par_arr = np.array([th for th in theta])

    if all(ii>=0 for ii in (par_arr - prl)) and all(ii>=0 for ii in (prh - par_arr)):

        return 0.

    return -np.inf


def lnlikelihood(theta,data,errors,mask,par_mask,switch):
    
    """
    CORE FUNCTION: lnlikelihood
    INPUTS: theta, data, errors, mask, par_mask, switch
    
    theta -- unpacks as a tuple of parameters, i.e. a location at which to evaluate the log-likelihood.
    data, errors -- the measured values of the data points and the estimated uncertainties
    mask, par_mask -- mask for the data-points and par_mask for the parameter list 
    (allows us to select certain datapoints and turn model parameters on/off)
    switch -- boolean, determines whether to return (log-likelihood, values) (TRUE) or just log-likelihood (FALSE)
    
    Note that this log-likelihood function evaluates the equivalent of a chi-squared. A number of other 'experimental'
    log-likelihood functions are available, with the same inputs unless otherwise stated.
    """
    values = bpl.predictor(par_mask, theta)
        
    if switch != 0:
        return -0.5*(np.sum( ((data[mask]-values[mask])/errors[mask])**2. )), values

    else:
        return -0.5*(np.sum( ((data[mask]-values[mask])/errors[mask])**2. ))

def lnprob(theta, data, errors, mask, param_mask, prl, prh):
    
    """
    CORE FUNCTION: lnprob
    INPUTS: theta, data, errors, mask, param_mask, prl, prh
    
    theta -- unpacks as a tuple of parameters, i.e. a location at which to evaluate the log-likelihood.
    data, errors -- the measured values of the data points and the estimated uncertainties
    mask, pararm_mask -- mask for the data-points and param_mask for the parameter list
    prl, prh -- label the low and high boundaries of the prior box.
    
    lnprob calls lnprior on the input (theta, prl, prh) and then calls lnlikelihood if within the boundaries of 
    the prior box.
    
    OUTPUTS: posterior_probability, values
    
    posterior_probability -- actually written as lnprior + lnlikelihood (so feel free to alter either function).
    Of course, this is = ln(prior*likelihood).
    values -- likelihood is called with the switch parameter = TRUE, so the predictions of the model are returned
    (this is important because it means we can save the distribution of predictions as well as the parameters)
    """
    
    lp = lnprior(theta, prl, prh)
    
    if not np.isfinite(lp): 
        return -np.inf, bpl.predictor(param_mask, theta)
    
    lnl, vals = lnlikelihood(theta,data,errors,mask,param_mask,1)
    
    return lp + lnl , vals

# second version for easy handling of experiments with the likelihood

def lnprob2(theta, data, errors, mask, param_mask, prl, prh, exp_lnl):
    
    """
    lnprob2 -- see lnprob for documentation.
    The difference is an additional input, exp_lnl, intended to be 
    one of the experimental likelihood functions (e.g. up_lnl) which 
    will be used in the evaluation of the log-probability.
    """
    
    lp = lnprior(theta, prl, prh)

    if not np.isfinite(lp): 
        return -np.inf, bpl.predictor(param_mask, theta)
    
    lnl, vals = exp_lnl(theta,data,errors,mask,param_mask,1)
    
    return lp + lnl , vals

### CONVENIENCE FUNCTIONS

def distrib_eval(distribution, prl, prh):
    
    """
    CONVENIENCE FUNCTION: distrib_eval
    INPUTS: distribution,  prl, prh
    
    Feed in a distribution (usually the posterior probability distribution of the model parameters,
    and technically the set of *draws* from the distribution) and the limits of the distribution
    (prl, prh) -- low and high limits -- which may be e.g. the prior box imposed on the posterior 
    distribution.
    
    This function uses kernel density estimation to robustly locate the principal mode of the 
    distribution (i.e. the highest peak, the single most likely value in parameter space). 
    The distribution is then split on either side of this peak and used to determine an
    assymetric 68% confidence interval. This is then used to evaluate +/- assymetric uncertainties.
    
    OUTPUTS: peak_values, siglo, sighi [modal values, low uncertainty, high uncertainty]
    """
    
    peak_values = []
    siglo = []
    sighi = []
    
    ndim = distribution.shape[1] # dimensionality of the distribution [N_draws, ndim]
    
    for ii in range(ndim):

        kde_curve = kde(distribution[:,ii]).evaluate(np.linspace(prl[ii],prh[ii],1000))

        result = np.mean(np.linspace(prl[ii],prh[ii],1000)[kde_curve==np.max(kde_curve)])

        peak_values.append(result)

        loval = np.percentile(distribution[:,ii][distribution[:,ii]<=result],32.)
        hival = np.percentile(distribution[:,ii][distribution[:,ii]>=result],68.)

        siglo.append(result-loval)
        sighi.append(hival-result)

    return peak_values, siglo, sighi

def limfunc(axarr,p_min,p_max):

    """
    This function takes a corner plot on axis grid 'axarr' and applies the lower and upper plotting
    limits given in p_min, p_max.
    """

    ndim = p_min.size

    for ii in range(ndim):
        for jj in range(ndim):

            if ii>=jj: axarr[ii*ndim+jj].set_xlim([p_min[jj],p_max[jj]])
            if ii>jj:  axarr[ii*ndim+jj].set_ylim([p_min[ii],p_max[ii]])


### CLASS DEFINITION FOR EMCEE
    
class emcee_wrapper(object):

    """ Yes, I am proud of myself for the pun."""

    def __init__(self, data,errors,mask, param_msk,priors_lo,priors_hi, identity):
        
        """ 
        INPUTS: 
        data, errors, mask, [define dataset]
        param_msk, priors_low,priors_high, [define edges of box]
        identity [name the dataset] 
        """

        self.data = data
        self.errors = errors
        self.mask = mask

        self.pmsk= param_msk
        self.prl = priors_lo[self.pmsk]
        self.prh = priors_hi[self.pmsk]
        
        self.identity = identity

        print self.identity+': MCMC object created'

    
    def MLEstRoutine(self):
        
        """ Method for maximum-likelihood estimation (MUCH faster than Bayesian approach, doesn't give you the posterior distribution). 
        Prints out a summary, makes available .MLE_result (optimised parameter values) and .MLE_predictions (the latter are the set of 
        predicted data implied by the optimised parameters)."""
        
        cube_centre = (self.prl+self.prh)/2.

        nll = lambda *args: -lnlikelihood(*args)

        self.MLE_result = opt.minimize(nll, list(cube_centre), args=(self.data,self.errors,self.mask,self.pmsk,0.))

        self.MLE_predictions = bpl.predictor(self.pmsk, self.MLE_result["x"])
        
        par_labels = np.array(["log(age)", "X1", " X2", "[Z/H]", "[Mg/Fe]", "[Ca/Fe]", "[Na/Fe]", "[K/Fe]", "[C/Fe]", "[Fe/Z]", "[Ti/Fe]", "$\Delta$T_eff", "[ONeS/Fe]"])[self.pmsk]

        feat_labels = np.array(['C$_2$','H$\\beta$','Fe5015', 'Mgb', 'FeI 0.52', 'FeI 0.53','NaI 0.59', 'NaI 0.82','CaII Triplet', 'MgI 0.88', 'FeH', 'CaI 1.06', 
                                'NaI 1.14', 'KI 1.17', 'KI 1.25', 'AlI 1.31', 'CaI 1.98', 'NaI 2.21', 'CaI 2.26', 'CO a','CO b'])[self.mask]


        print 'FEATURES: ', feat_labels
        print 'Data: ', self.data[self.mask]
        print 'Predictions: ', self.MLE_predictions[self.mask]
        print 'Residuals: ', self.data[self.mask]-bpl.predictor(self.pmsk, self.MLE_result["x"])[self.mask]
        print '            '

        print 'PARAMETERS: ', par_labels
        print 'RESULT: ',self.MLE_result["x"]
        print '            '
    
        print 'Chi-squared: ', np.sum( ((self.data[self.mask]-bpl.predictor(self.pmsk, self.MLE_result["x"])[self.mask])/(self.errors[self.mask]))**2. )

        print '           '

    def Runner(self,nwalkers,nsteps,nburn):
        
        """ 
        INPUT: N_Walkers, N_steps, N_burn_in_steps
        Method runs an MCMC chain according to inputs.
        """

        self.ndim = self.prl.size
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lnprob, args=(self.data,self.errors,self.mask,self.pmsk,self.prl,self.prh))
        
        print self.identity+': sampler created'

        p_min = self.prl
        p_max = self.prh

        psiz = p_max - p_min

        np.random.seed(5) # for repeatability

        pos = [p_min + psiz*np.random.rand(self.ndim) for ii in range(self.nwalkers)]
           
        pos, prob, state, metadata = sampler.run_mcmc(pos,self.nsteps)

        self.samples = sampler.chain[:, nburn:, :].reshape((-1, self.ndim))

        self.chainvals = np.array(sampler.blobs)[nburn:,:,self.mask].reshape((-1,len(self.data[self.mask]))) # use metadata method to store values
               
        print self.identity+': MCMC run complete'
        
        ### save for convergence testing

        self.chain = sampler.chain

        self.all_samples = sampler.chain[:, 1:, :].reshape((-1, self.ndim))
        self.samples_1 = sampler.chain[:, 50:, :].reshape((-1, self.ndim))
        self.samples_2 = sampler.chain[:, 100:, :].reshape((-1, self.ndim))
        self.samples_3 = sampler.chain[:, 150:, :].reshape((-1, self.ndim))
        self.samples_4 = sampler.chain[:, 500:, :].reshape((-1, self.ndim))

    def DisplayResults(self):
        
        """
        Display the results of the MCMC run.
        Window 1: parameter corner plot
        Window 2: reconstructed EWs vs. actual EWs

        Evaluates the best-fit value through a KDE method and the (potentially skewed) 68% confidence
        interval boundaries.

        Also evaluates the predicted values given the best fit results (self.predictions)
        """

        results_tuple = map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),
                    zip(*np.percentile(self.samples, [16,50,84],axis=0))
                    )
        
        self.marginalised_results = results_tuple
        
        self.peak_values = []
        self.siglo = []
        self.sighi = []

        for ii in range(self.ndim):

            kde_curve = kde(self.samples[:,ii]).evaluate(np.linspace(self.prl[ii],self.prh[ii],1000)) # needs to be fine grained even if it takes longer (peak resolution matters)
            result = np.mean(np.linspace(self.prl[ii],self.prh[ii],1000)[kde_curve==np.max(kde_curve)])
            
            self.peak_values.append(result)
        

            if self.samples[:,ii][self.samples[:,ii]<=result].size >0: 
                self.siglo.append( np.percentile(self.samples[:,ii][self.samples[:,ii]<=result],32.) ) # split into two distributions on either side of the peak. 
            else:
                self.sighi.append(0.) # hopefully fine grained KDE avoids this, but just in case peak=prior bound, must have robust exit strategy

            if self.samples[:,ii][self.samples[:,ii]>=result].size >0:
                self.sighi.append( np.percentile(self.samples[:,ii][self.samples[:,ii]>=result],68.) ) # take 68th percentile on either side; guaranteed to enclose 68% total.
            else:
                self.sighi.append(0.)
        

        headline_results = [results_tuple[rr][0] for rr in range(self.ndim)]

        par_labels = np.array(["log(age)", "X1", " X2", "[Z/H]", "[Mg/Fe]", "[Ca/Fe]", "[Na/Fe]", "[K/Fe]", "[C/Fe]", "[Fe/Z]", "[Ti/Fe]", "$\Delta$T_eff", "[ONeS/Fe]"])[self.pmsk]

        if hasattr(self,"MLE_result"): 
            self.f1 = corner.corner(self.samples, labels=par_labels, truths=list(self.MLE_result["x"]), quantiles=[0.16, 0.5, 0.84], truth_color='r')
        else: 
            self.f1 = corner.corner(self.samples, labels=par_labels, truths=self.peak_values, quantiles=[0.16, 0.5, 0.84], truth_color='r')
                
        self.f1.savefig(plot_dir+'self.identity+'_MCMC.png',orientation='landscape')

        msk_label = np.array(['C$_2$','H$\\beta$','Fe5015', 'Mgb', 'FeI 0.52', 'FeI 0.53','NaI 0.59', 'NaI 0.82','CaII Triplet', 'MgI 0.88', 'FeH', 'CaI 1.06', 
                                'NaI 1.14', 'KI 1.17', 'KI 1.25', 'AlI 1.31', 'CaI 1.98', 'NaI 2.21', 'CaI 2.26', 'CO a', 'CO b'])[self.mask]

        ms_data = self.data[self.mask]
        ms_errors=self.errors[self.mask]


        self.peak_prediction = bpl.predictor(self.pmsk, self.peak_values)
        
        self.prediction = np.percentile(self.chainvals,[50],axis=0)[0] # [0] so (1,N) --> (N)
        
        self.peak_values = np.array(self.peak_values)
        self.siglo = np.array(self.siglo)
        self.sighi = np.array(self.sighi)

        self.results = np.vstack((self.peak_values,self.peak_values-self.siglo,self.sighi-self.peak_values))

        self.dof = self.data[self.mask].size - self.peak_values.size # N_data - N_param

        self.r_chi_s = np.sum( ((self.data[self.mask] - self.prediction)**2)/(self.errors[self.mask]**2.))/self.dof # reduced chi^2

    def ComputeResults(self):
        
        """
        As .DisplayResults() but without plotting output.

        """

        results_tuple = map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),
                            zip(*np.percentile(self.samples, [16,50,84],axis=0))
                            )
        
        self.marginalised_results = results_tuple
        
        self.peak_values = []
        self.siglo = []
        self.sighi = []

        for ii in range(self.ndim):

            kde_curve = kde(self.samples[:,ii]).evaluate(np.linspace(self.prl[ii],self.prh[ii],1000)) # needs to be fine grained even if it takes longer (peak resolution matters)
            result = np.mean(np.linspace(self.prl[ii],self.prh[ii],1000)[kde_curve==np.max(kde_curve)])
            
            self.peak_values.append(result)
        

            if self.samples[:,ii][self.samples[:,ii]<=result].size >0: 
                self.siglo.append( np.percentile(self.samples[:,ii][self.samples[:,ii]<=result],32.) ) # split into two distributions on either side of the peak. 
            else:
                self.sighi.append(0.) # hopefully fine grained KDE avoids this, but just in case peak=prior bound, must have robust exit strategy

            if self.samples[:,ii][self.samples[:,ii]>=result].size >0:
                self.sighi.append( np.percentile(self.samples[:,ii][self.samples[:,ii]>=result],68.) ) # take 68th percentile on either side; guaranteed to enclose 68% total.
            else:
                self.sighi.append(0.)
        

        headline_results = [results_tuple[rr][0] for rr in range(self.ndim)]

        par_labels = np.array(["log(age)", "X1", " X2", "[Z/H]", "[Mg/Fe]", "[Ca/Fe]", "[Na/Fe]", "[K/Fe]", "[C/Fe]", "[Fe/Z]", "[Ti/Fe]", "$\Delta$T_eff", "[ONeS/Fe]"])[self.pmsk]

        msk_label = np.array(['C$_2$','H$\\beta$','Fe5015','Mgb', 'FeI 0.52', 'FeI 0.53','NaI 0.59', 'NaI 0.82','CaII Triplet', 'MgI 0.88', 'FeH', 'CaI 1.06', 
                                'NaI 1.14', 'KI 1.17', 'KI 1.25', 'AlI 1.31', 'CaI 1.98', 'NaI 2.21', 'CaI 2.26', 'CO a', 'CO b'])[self.mask]

        ms_data = self.data[self.mask]
        ms_errors=self.errors[self.mask]


        self.peak_prediction = bpl.predictor(self.pmsk, self.peak_values)
        
        self.prediction = np.percentile(self.chainvals,[50],axis=0) # [0] so (1,N) --> (N)

        self.peak_values = np.array(self.peak_values)
        self.siglo = np.array(self.siglo)
        self.sighi = np.array(self.sighi)

        self.results = np.vstack((self.peak_values,self.peak_values-self.siglo,self.sighi-self.peak_values))

        self.dof = self.data[self.mask].size - self.peak_values.size # N_data - N_param

        self.r_chi_s = np.sum( ((self.data[self.mask] - self.prediction)**2)/(self.errors[self.mask]**2.))/self.dof # reduced chi^2
        
        
    def DisplayPredictions(self):

        """ Displays the distribution of predicted data derived from the posterior probability distribution."""
        msk_label = np.array(['C$_2$','H$\\beta$','Fe5015', 'Mgb', 'FeI 0.52', 'FeI 0.53','NaI 0.59', 'NaI 0.82','CaII Triplet', 'MgI 0.88', 'FeH', 'CaI 1.06', 
                                'NaI 1.14', 'KI 1.17', 'KI 1.25', 'AlI 1.31', 'CaI 1.98', 'NaI 2.21', 'CaI 2.26', 'CO a', 'CO b'])[self.mask]

        self.cv_fig = corner.corner(self.chainvals,
                                    labels=msk_label,color='r',quantiles=[.16,.50,.84],alpha=0.5,
                                    hist_kwargs={"linewidth": 3},label_kwargs={"fontsize":'x-large'},
                                    plot_datapoints=False,plot_density=False)
        
        axarr = self.cv_fig.get_axes()

        dat = self.data[self.mask]
        err = self.errors[self.mask]

        for ii in range(dat.size):
            for jj in range(dat.size):

                if ii==jj: 
                
                    axarr[ii+dat.size*jj].axvline(dat[ii],color='r')
                    axarr[ii+dat.size*jj].axvspan(dat[ii]-err[ii],dat[ii]+err[ii],alpha=0.3,color='r')

                if ii<jj: axarr[ii+dat.size*jj].errorbar(dat[ii],dat[jj],xerr=err[ii],yerr=err[jj],fmt='s',ms=10,color='r')

        self.cv_fig.savefig(plot_dir+'self.identity+'_MCMC_recon.png',orientation='landscape')

    def ConvergenceTest(self):

        """This method is intended to help the user judge whether the chosen burn-in was appropriate (i.e. had the MCMC walkers converged by the step at which the cutoff
        was applied). NB as of v1.0 the plotting is not generalised for arbitrary parameter choice and lacks error handling (e.g. to check whether the chain has actually
        been run). It is usually obvious from the results plots whether convergence was achieved or not..."""
        
        f2,ax2 = plt.subplots(5,2)

        par_labels = np.array(["log(age)", "X1", " X2", "[Z/H]", "[Mg/Fe]", "[Ca/Fe]", "[Na/Fe]", "[K/Fe]", "[C/Fe]", "[Fe/Z]", "[Ti/Fe]", "$\Delta$T_eff", "[ONeS/Fe]"])[self.pmsk]

        for kk in range(self.ndim):
            for jj in range(self.nwalkers): 

                ax2.flat[kk].plot(np.arange(self.nsteps),self.chain[jj,:,kk],'k-',alpha=0.3)
                ax2.flat[kk].set_title(par_labels[kk],fontsize=20)

        raw_input('press enter to proceed to results plot')

        vals0, slo0, shi0 = distrib_eval(self.all_samples,self.ndim,self.prl,self.prh)
        vals1, slo1, shi1 = distrib_eval(self.samples_1,self.ndim,self.prl,self.prh)
        vals2, slo2, shi2 = distrib_eval(self.samples_2,self.ndim,self.prl,self.prh)
        vals3, slo3, shi3 = distrib_eval(self.samples_3,self.ndim,self.prl,self.prh)
        vals4, slo4, shi4 = distrib_eval(self.samples,  self.ndim,self.prl,self.prh)
        vals5, slo5, shi5 = distrib_eval(self.samples_4,self.ndim,self.prl,self.prh)

        cuts = np.array([1,50,100,150,200,500])

        f,axr = plt.subplots(5,2) # if 9 parameters...

        for ii in range(self.ndim):

            ax = axr.flat[ii]

            ax.plot(cuts[0],vals0[ii],'ko')
            ax.plot(cuts[1],vals1[ii],'ko')
            ax.plot(cuts[2],vals2[ii],'ko')
            ax.plot(cuts[3],vals3[ii],'ko')
            ax.plot(cuts[4],vals4[ii],'ko')
            ax.plot(cuts[5],vals5[ii],'ko')

            ax.plot([cuts[0],cuts[0]],[vals0[ii]-slo0[ii],vals0[ii]+shi0[ii]],'k:')
            ax.plot([cuts[1],cuts[1]],[vals1[ii]-slo1[ii],vals1[ii]+shi1[ii]],'k:')
            ax.plot([cuts[2],cuts[2]],[vals2[ii]-slo2[ii],vals2[ii]+shi2[ii]],'k:')
            ax.plot([cuts[3],cuts[3]],[vals3[ii]-slo3[ii],vals3[ii]+shi3[ii]],'k:')
            ax.plot([cuts[4],cuts[4]],[vals4[ii]-slo4[ii],vals4[ii]+shi4[ii]],'k:')
            ax.plot([cuts[5],cuts[5]],[vals5[ii]-slo5[ii],vals5[ii]+shi5[ii]],'k:')

        raw_input('press enter to continue')


    def OutputTable(self,output=True):

        """
        Method for output of a results table. If keyword output=True (default setting) then a .fits table
        is read out to results/. Either way, a 
        structured array with name, reduced chi squared, parameter values and errors is stored.
        """

        outstruct = np.zeros( 1, dtype=[('NAME','20S'),('rcs',float)])

        outstruct['NAME'] = self.identity

        outstruct['rcs']  = self.r_chi_s

        taglist= np.array(["lage", "X1", "X2", "Z", "mg", "ca", "na", "k", "c", "fe", "ti", "teff", "ones"])[self.pmsk]

        for xx in range(self.results.shape[1]):

            tag = taglist[xx]

            outstruct = append_fields(outstruct,tag,np.ones(1)*self.results[0,xx], usemask=False) # modal parameter values

            outstruct = append_fields(outstruct,tag+'_em',np.ones(1)*self.results[1,xx], usemask=False) # asymmetric errors: minus
            outstruct = append_fields(outstruct,tag+'_ep',np.ones(1)*self.results[2,xx], usemask=False) # plus

        self.outstruct = outstruct # store results structure

        if output: 

            fits.writeto(res_dir+self.identity+'_results.fits',outstruct,clobber=True)
            
            np.save(res_dir+self.identity+'_predictions.npy',self.prediction) # also save the best-fit predictions of the model
            np.save(res_dir+self.identity+'_par_samples.npy',self.samples)    # also save the chains themselves


    def ExperimentalRunner(self,nwalkers,nsteps,nburn,exp_lnl):
        
        """ 
        INPUT: N_Walkers, N_steps, N_burn_in_steps, experimental_loglikelihood_function
        Method runs an MCMC chain according to inputs, using experimental likelihood function.
        (otherwise precisely the same as standard .Runner() method)
        """

        self.ndim = self.prl.size
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lnprob2, args=(self.data,self.errors,self.mask,self.pmsk,self.prl,self.prh, exp_lnl))
        
        print self.identity+': sampler created, experimental likelihood function being used.'

        p_min = self.prl
        p_max = self.prh

        psiz = p_max - p_min

        np.random.seed(5) # for repeatability

        pos = [p_min + psiz*np.random.rand(self.ndim) for ii in range(self.nwalkers)]
           
        pos, prob, state, metadata = sampler.run_mcmc(pos,self.nsteps)

        self.samples = sampler.chain[:, nburn:, :].reshape((-1, self.ndim))

        self.chainvals = np.array(sampler.blobs)[nburn:,:,self.mask].reshape((-1,len(self.data[self.mask]))) # use metadata method to store values
               
        print self.identity+': MCMC run complete'
        
        ### save for convergence testing

        self.chain = sampler.chain

        self.all_samples = sampler.chain[:, 1:, :].reshape((-1, self.ndim))
        self.samples_1 = sampler.chain[:, 50:, :].reshape((-1, self.ndim))
        self.samples_2 = sampler.chain[:, 100:, :].reshape((-1, self.ndim))
        self.samples_3 = sampler.chain[:, 150:, :].reshape((-1, self.ndim))
        self.samples_4 = sampler.chain[:, 500:, :].reshape((-1, self.ndim))




### PLOTTING METHODS CONVERTED TO FUNCTIONS FOR SAVED CHAINS/DISTRIBUTIONS OF PREDICTIONS:

def DisplayResults(samples,save=False):
        
    """
    Display the results of the MCMC run given input samples.
    
    Evaluates the best-fit value through a KDE method and the (potentially skewed) 68% confidence
    interval boundaries.

    Also evaluates the predicted values given the best fit results (predictions)
    """

    results_tuple = map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),
                        zip(*np.percentile(samples, [16,50,84],axis=0))
                        )
        
    marginalised_results = results_tuple
        
    peak_values = []
    siglo = []
    sighi = []

    for ii in range(ndim):

        kde_curve = kde(samples[:,ii]).evaluate(np.linspace(prl[ii],prh[ii],1000)) # needs to be fine grained even if it takes longer (peak resolution matters)
        result = np.mean(np.linspace(prl[ii],prh[ii],1000)[kde_curve==np.max(kde_curve)])
        
        peak_values.append(result)
        

        if samples[:,ii][samples[:,ii]<=result].size >0: 
            siglo.append( np.percentile(samples[:,ii][samples[:,ii]<=result],32.) ) # split into two distributions on either side of the peak. 
        else:
            sighi.append(0.) # hopefully fine grained KDE avoids this, but just in case peak=prior bound, must have robust exit strategy
                
        if samples[:,ii][samples[:,ii]>=result].size >0:
            sighi.append( np.percentile(samples[:,ii][samples[:,ii]>=result],68.) ) # take 68th percentile on either side; guaranteed to enclose 68% total.
        else:
            sighi.append(0.)
        
    headline_results = [results_tuple[rr][0] for rr in range(ndim)]

    par_labels = np.array(["log(age)", "X1", " X2", "[Z/H]", "[Mg/Fe]", "[Ca/Fe]", "[Na/Fe]", "[K/Fe]", "[C/Fe]", "[Fe/Z]", "[Ti/Fe]", "$\Delta$T_eff", "[ONeS/Fe]"])[pmsk]
    
    f1 = corner.corner(samples, labels=par_labels, truths=peak_values, quantiles=[0.16, 0.5, 0.84])
    
    if save: f1.savefig(plot_dir+'identity+'_MCMC.png',orientation='landscape')

    msk_label = np.array(['C$_2$','H$\\beta$','Fe5015', 'Mgb', 'FeI 0.52', 'FeI 0.53','NaI 0.59', 'NaI 0.82','CaII Triplet', 'MgI 0.88', 'FeH', 'CaI 1.06', 
                          'NaI 1.14', 'KI 1.17', 'KI 1.25', 'AlI 1.31', 'CaI 1.98', 'NaI 2.21', 'CaI 2.26', 'CO a', 'CO b'])[mask]

    ms_data = data[mask]
    ms_errors=errors[mask]


    peak_prediction = bpl.predictor(pmsk, peak_values)
        
    prediction = np.percentile(chainvals,[50],axis=0)[0] # [0] so (1,N) --> (N)
        
    peak_values = np.array(peak_values)
    siglo = np.array(siglo)
    sighi = np.array(sighi)

    results = np.vstack((peak_values,peak_values-siglo,sighi-peak_values))

    dof = data[mask].size - peak_values.size # N_data - N_param

    r_chi_s = np.sum( (data[mask] - prediction)**2/errors[mask]**2.)/dof # reduced chi^2
        
    return f1
    
def DisplayPredictions(samples,par_mask, data,errors,mask, save=False):

    """
    Display the distribution of predictions saved from an MCMC run.

    INPUTS: 

    samples  -- saved MCMC chain
    par_mask -- parameter selection mask used to create the chain

    data -- the original data the model was fit too
    errors -- the errors on that data
    mask -- feature selection mask used (or not used!) to create the chain

    KEYWORD:
    save=False -- if true, save the plot

    RETURNS:
    figure object

    """

    full_samples = np.zeros( (samples.shape[0],13),float )
    full_samples[:,par_mask] = samples
    chainvals = bpl.fast_predictor(full_samples)[:,mask]


    msk_label = np.array(['C$_2$','H$\\beta$','Fe5015', 'Mgb', 'FeI 0.52', 'FeI 0.53','NaI 0.59', 'NaI 0.82','CaII Trip.', 'MgI 0.88', 'FeH', 'CaI 1.06', 
                          'NaI 1.14', 'KI 1.17', 'KI 1.25', 'AlI 1.31', 'CaI 1.98', 'NaI 2.21', 'CaI 2.26', 'CO a', 'CO b'])[mask]

    cv_fig = corner.corner(chainvals,
                           labels=msk_label,color='r',quantiles=[.16,.50,.84],alpha=0.5,
                           hist_kwargs={"linewidth": 3},label_kwargs={"fontsize":'large'},
                           plot_datapoints=False,plot_density=False)
        
    axarr = cv_fig.get_axes()

    dat = data[mask]
    err = errors[mask]

    for ii in range(dat.size):
        for jj in range(dat.size):

            if ii==jj: 
                
                axarr[ii+dat.size*jj].axvline(dat[ii],color='r')
                axarr[ii+dat.size*jj].axvspan(dat[ii]-err[ii],dat[ii]+err[ii],alpha=0.3,color='r')

            if ii<jj: axarr[ii+dat.size*jj].errorbar(dat[ii],dat[jj],xerr=err[ii],yerr=err[jj],fmt='s',ms=10,color='r')

    if save: cv_fig.savefig(plot_dir+'identity+'_MCMC_recon.png',orientation='landscape')
    
    return cv_fig



def ChainConverter(samples, par_mask):

    """
    INPUTS: samples (pre-computed MCMC chain), par_mask (to inform ChainConverter which optional parameters are missing from the chain)
    
    This function is just some witchcraft (not really) to convert the model parameters into a reorganised form more comparable with the 
    astronomy literature. For example, we usually want to know [Mg/Fe], rather than [Mg/Z]. This entails subtracting [Fe/Z] to get [Mg/Fe]
    *relative to the library stars with metallicity [Z/H]*. We actually want [Mg/Fe] relative to the solar metallicity ([Z/H]=0), so a set 
    of empirical corrections are applied to some elements (these are the recommended choices for the CvD-16 models).
    
    In addition, the pair of IMF slopes X1, X2 are converted into f_dwarf and (X2-X1) (dwarf enrichment, IMF shape proxy). In v1.0 we do 
    assume you are fitting for X1 AND X2. If you aren't, better to insert a column of fixed values into 'samples' in the correct place 
    rather than fiddling with this function.
    """
    
    zvals = np.array([-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2])

    ofe = np.array([0.6,0.5,0.5,0.4,0.3,0.2,0.2,0.1,0.0,0.0])
    mgfe= np.array([0.4,0.4,0.4,0.4,0.34,0.22,0.14,0.11,0.05,0.04])
    cafe= np.array([0.32,0.3,0.28,0.26,0.26,0.17,0.12,0.06,0.0,0.0])

    ######

    tags = np.array(['lage','X1','X2','Z','mg','ca','na','k','cbn','fe','ti','Teff','ONeS'])[par_mask]

    new_chains = np.zeros( (samples.shape[0],tags.size), float)

    for tt,tag in enumerate(tags):
        if tag=='Z':  Z_loc = tt
        if tag=='fe': fe_loc = tt

    print 

    for tt,tag in enumerate(tags):

        if tag=='lage': new_chains[:,tt] = np.exp(1.)**samples[:,tt] # convert to age in Gyr
        elif tag=='X1':   new_chains[:,tt] = fd_find(samples[:,tt],samples[:,tt+1]) # calculate f_dwarf distribution
        elif tag=='X2':   new_chains[:,tt] = samples[:,tt] - samples[:,tt-1] # calculate X2-X1 (shape proxy) distribution
        elif tag=='Z':    new_chains[:,tt] = samples[:,tt]
        elif tag=='fe':   new_chains[:,tt] = samples[:,tt] + samples[:,Z_loc]

        elif tag=='mg':   new_chains[:,tt] = samples[:,tt] - samples[:,fe_loc] + np.interp(samples[:,Z_loc],zvals,mgfe)
        elif tag=='ca':   new_chains[:,tt] = samples[:,tt] - samples[:,fe_loc] + np.interp(samples[:,Z_loc],zvals,cafe)
        elif tag=='ONeS': new_chains[:,tt] = samples[:,tt] - samples[:,fe_loc] + np.interp(samples[:,Z_loc],zvals,ofe)

        elif tag=='Teff': print 'WARNING: THIS FUNCTION IS ONLY TO BE USED WITH THE VILLAUME UPDATE TO THE MODELS (CvD-16)'

        else: new_chains[:,tt] = samples[:,tt] - samples[:,fe_loc]

    par_labels = np.array(["age /Gyr", "f$_{\\mathrm{dwarf}}$", "X2-X1", "[Z/H]", "[Mg/Fe]", "[Ca/Fe]", "[Na/Fe]", "[K/Fe]", "[C/Fe]", "[Fe/H]", "[Ti/Fe]", "$\Delta$T_eff", "[O,Ne,S/Fe]"])[par_mask]

    return new_chains, par_labels


def ChainConverter2(samples, par_mask):

    """
    This version of ChainConverter assumes you're drawing from {f_dwarf, (X2-X1)} rather than {X1,X2}

    """


    zvals = np.array([-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2])

    ofe = np.array([0.6,0.5,0.5,0.4,0.3,0.2,0.2,0.1,0.0,0.0])
    mgfe= np.array([0.4,0.4,0.4,0.4,0.34,0.22,0.14,0.11,0.05,0.04])
    cafe= np.array([0.32,0.3,0.28,0.26,0.26,0.17,0.12,0.06,0.0,0.0])

    ######

    tags = np.array(['lage','X1','X2','Z','mg','ca','na','k','cbn','fe','ti','Teff','ONeS'])[par_mask]

    new_chains = np.zeros( (samples.shape[0],tags.size), float)

    for tt,tag in enumerate(tags):
        if tag=='Z':  Z_loc = tt
        if tag=='fe': fe_loc = tt

    print 

    for tt,tag in enumerate(tags):

        if tag=='lage': new_chains[:,tt] = np.exp(1.)**samples[:,tt] # convert to age in Gyr
        elif tag=='X1':   new_chains[:,tt] = samples[:,tt]
        elif tag=='X2':   new_chains[:,tt] = samples[:,tt]
        elif tag=='Z':    new_chains[:,tt] = samples[:,tt]
        elif tag=='fe':   new_chains[:,tt] = samples[:,tt] + samples[:,Z_loc]

        elif tag=='mg':   new_chains[:,tt] = samples[:,tt] - samples[:,fe_loc] + np.interp(samples[:,Z_loc],zvals,mgfe)
        elif tag=='ca':   new_chains[:,tt] = samples[:,tt] - samples[:,fe_loc] + np.interp(samples[:,Z_loc],zvals,cafe)
        elif tag=='ONeS': new_chains[:,tt] = samples[:,tt] - samples[:,fe_loc] + np.interp(samples[:,Z_loc],zvals,ofe)

        elif tag=='Teff': print 'WARNING: THIS FUNCTION IS ONLY TO BE USED WITH THE VILLAUME UPDATE TO THE MODELS'

        else: new_chains[:,tt] = samples[:,tt] - samples[:,fe_loc]

    par_labels = np.array(["age /Gyr", "f$_{\\mathrm{dwarf}}$", "X2-X1", "[Z/H]", "[Mg/Fe]", "[Ca/Fe]", "[Na/Fe]", "[K/Fe]", "[C/Fe]", "[Fe/H]", "[Ti/Fe]", "$\Delta$T_eff", "[O,Ne,S/Fe]"])[par_mask]

    return new_chains, par_labels


def GetParams(file_name):

    """INPUT: file_name
    Grabs the results from a saved results file and reads in the most probable theta values from it."""

    results = fits.getdata(file_name)

    tags = np.array(['lage','X1', 'X2', 'Z', 'na', 'mg', 'ca', 'ti', 'fe', 'c', 'k', 'ones'])

    def_vals = np.array([np.log(13.5),1.3,2.3,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    params = []

    for tag in tags:
        if tag in results.dtype.names:
            params.append(results[tag])
        elif tag=='X2': # fix the single power-law case
            params.append(results['X1'])
        else:
            params.append(def_vals[tags==tag])

    return np.array(params)
