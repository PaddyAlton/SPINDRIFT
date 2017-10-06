# BPL_index_model.py
# universally callable - upgraded version of index_model with capacity to call
# a two-part IMF.
# PaddyAlton -- 2017-03-01

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white"); sns.set_color_codes(palette='colorblind')
import astropy.io.fits as fits
import scipy.optimize as opt

import glob

from scipy.interpolate import UnivariateSpline as usplin
from scipy.interpolate import RectBivariateSpline as rbsplin
from scipy.interpolate import RegularGridInterpolator as rg_int

### DEFINE A PLOTTING FUNCTION FOR LATER USE

def gridgrad(xvals,yvals,axis,c0="Blues",c1="Oranges",**kwargs):
    
    """
    gridgrad()

    USE: plotting a 2D grid of x-values against a 2D grid of y-values and using a 2-colour 
    space to show position within the grid.

    """
    
    s0 = xvals.shape[0]
    s1 = xvals.shape[1]
        
    cols0 = sns.color_palette(c0,s0)
    cols1 = sns.color_palette(c1,s1)
    
    for kk in range(s0): axis.plot(xvals[kk,:],yvals[kk,:],'-',c=cols0[kk],linewidth=3,**kwargs)
    for kk in range(s1): axis.plot(xvals[:,kk],yvals[:,kk],'-',c=cols1[kk],linewidth=3,**kwargs)
    
    for ii in range(s0):
        for jj in range(s1):
            axis.plot(xvals[ii,jj],yvals[ii,jj],'k.',**kwargs)

### CONVENIENCE FUNCTIONS ###

def splintestfunc(vals,fid_vals,grid,splines,corr_list,titles):

    """ This function was of use in confirming that the model smoothly and accurately interpolates the grid. """
    f,axs=plt.subplots(5,4)
    f.set_tight_layout(True)

    for ii in range(20):

        ax = axs.flat[ii]

        ax.plot(vals,(grid[:,ii]+fid_vals[ii])/fid_vals[ii],'ko')
        
        ax.plot(np.arange(11)/10.-0.5, (splines[ii](np.arange(11)/10.-0.5)+fid_vals[ii])/fid_vals[ii],'r-')
        
        ax.plot(np.arange(11)/10.-0.5, (splines[ii](np.arange(11)/10.-0.5)-corr_list[ii]+fid_vals[ii])/fid_vals[ii],'m-')

        ax.set_title(titles[ii],fontsize=22)

        if ax.get_ylim()[1]<1.05: ax.set_ylim(ax.get_ylim()[0],1.05)
        if ax.get_ylim()[0]>0.95: ax.set_ylim(0.95,ax.get_ylim()[1])
            
        ax.tick_params(labelsize=20)

def flexi_splintestfunc(vals,fid_vals,grid,splines,corr_list,titles,axs,**kwargs):

    """ As splintestfunc, but onto a custom axis object (axs) and with any plotting keywords desired (**kwargs). """
    for ii in range(20):

        ax = axs.flat[ii]

        ax.plot(vals,(grid[:,ii]+fid_vals[ii])/fid_vals[ii],'o',alpha=0.5)
        
        ax.plot(np.arange(21)/10.-0.5, (splines[ii](np.arange(21)/10.-0.5)+fid_vals[ii])/fid_vals[ii],'-')

        ax.set_title(titles[ii],fontsize=22)

        if ax.get_ylim()[1]<1.05: ax.set_ylim(ax.get_ylim()[0],1.05)
        if ax.get_ylim()[0]>0.95: ax.set_ylim(0.95,ax.get_ylim()[1])
            
        ax.tick_params(labelsize=20)

def fid_subtract(grid, ix):

    """
    Just use loops, it only has to be run once...

    'grid' is to become a grid of index variations W.R.T. X/Fe @(Z/H,X/Fe). We want to make it this
    by subtracting the fiducial value @Z/H, indexed by 'ix'.
    """

    for nn in range(grid.shape[2]): # loop over indices
        for ii in range(grid.shape[0]): # loop over metallicities

            fval = np.copy(grid[ii,ix,nn]) # set fiducial value of index nn at metallicity ii

            grid[ii,:,nn] = grid[ii,:,nn] - fval
                
    return grid

                

### CLASS DEFINITION ###

class modelset(object):

    def __init__(self):

        """
        Initialisation reads in a set of model index values. 

        Creates .basegrid(), array comprising: [21 indices, 16 X1 vals, 16 X2 vals, N Ages, M Metallicities]

        """
        
        mdirec = 'speclibrary/CVD_2PL/'
        
        error_message = "N.B. -- in order for SPINDRIFT to work, you need to compute the appropriate grids of spectroscopic index values, evaluated at sigma=230km/s, which we're subsequently going to interpolate over. These would then go in the speclibrary/CVD_2PL/ directory. I can provide these if you aren't repurposing the code, but would want permission from the CvD team whose work these grids are derived from. In this case drop me an email at paddyalton@gmail.com"
        if not os.path.isdir(mdirec): raise ValueError(error_message)
        
        self.modfil1 = mdirec+'indices230_solar_t13.5.fits'
        self.index_names = fits.getdata(self.modfil1)['index name']
        self.fid_grid = fits.getdata(self.modfil1)['EQW'] # 20 indices, 16x16 grid of underlying IMFs.
        self.fid_vals = self.fid_grid[:,4,9] # MW-like IMF predictions

        self.nind = self.index_names.size
        self.nimf = self.fid_grid.shape[1]

        self.imfx = 0.5 + np.arange(self.nimf)/5. # evaluated values of X1, X2 (slopes below/above 0.5M_solar)

        file_list = sorted(glob.glob(mdirec+'indices230*.fits'))

        self.ages_id = np.unique(np.array([xx.split('_')[-1] for xx in file_list]))
        self.met_id  = np.unique(np.array([xx.split('_')[-2] for xx in file_list]))[[3,2,1,4,0]] # hard code re-indexing of metallicities (because I foolishly altered the sensible filename structure... *sigh*)
        
        self.full_basegrid = np.zeros( (self.nind, self.nimf, self.nimf, self.ages_id.size, self.met_id.size) , float )

        for fil in file_list:

            ageid = fil.split('_')[-1]
            metid = fil.split('_')[-2]
            
            dat = fits.getdata(fil)['EQW']

            ix1 = np.where(self.ages_id==ageid)[0][0]
            ix2 = np.where(self.met_id==metid)[0][0]

            self.full_basegrid[:,:,:,ix1,ix2] = dat

        self.lages = np.log(np.array([7.,9.,11.,13.5])) # log(ages)
        self.zvals = np.array([-1.5,-1.1,-0.5,0.0,0.2]) # metallicity values

    def GordianSolution(self):
        
        """
        At the cost of smoothness, this implementation solves our problems by just allowing 4D
        linear interpolation of the grid.

        Creates .lingridfitter, a list of linear grid interpolations (one per index) 
        which should be called as self.lingridfitter[idx]((X1,X2,log(age),[Z/H]))

        This even supports extrapolation, which might matter for [Z/H].

        """
        
        self.lingridfitter = []

        for ii in range(self.nind): self.lingridfitter.append(rg_int( (self.imfx,self.imfx,self.lages,self.zvals), self.full_basegrid[ii], bounds_error=False,fill_value=None))
        

    def BivarAbundanceGrid(self):

        """
        Model Procedure: call specific values of log(age),X1,X2,Z to get the base grid. THEN call {Z,X/Fe} to get each abundance-caused
        *variation* on top of it. This method generates a set of bivariate grids over {Z, X} for each element of interest X. The underlying assumption
        is that the individual element abundance-based spectral deviations are independent of variations in parameters other than Z. This is reasonable 
        so long as you confine yourself to old stellar populations. Large IMF variations might change things at some level, e.g. La Barbera's team examine
        this possibility for [Na/Fe] + IMF variations, but the full modelling required for a general approach hasn't been done. Just bear it in mind as 
        one of a host of reasons you might not expect everything to line up perfectly in the infinite signal-to-noise limit.

        BivarAbundanceGrid creates the delta-eqw grid for each of a set of elements for five different values of Z (i.e. the deviations from the basic model
        *at* Z).

        """

        mdirec = 'speclibrary/CVD_2PL/'

        gridlist = sorted(glob.glob(mdirec+'*_response_grid.fits'))

        self.na_grid   = fid_subtract(fits.getdata(gridlist[5]),2) # load pre-computed extrapolated grid, subtract fiducial points.
        self.mg_grid   = fid_subtract(fits.getdata(gridlist[4]),2)
        self.ti_grid   = fid_subtract(fits.getdata(gridlist[8]),2)
        self.ca_grid   = fid_subtract(fits.getdata(gridlist[0]),2)
        self.fe_grid   = fid_subtract(fits.getdata(gridlist[2]),2)
        self.cbn_grid  = fid_subtract(fits.getdata(gridlist[1]),2)
        self.k_grid    = fid_subtract(fits.getdata(gridlist[3]),1)
        self.ones_grid = fid_subtract(fits.getdata(gridlist[6]),1)
        self.teff_grid = fid_subtract(fits.getdata(gridlist[7]),2)



    def BivarAbundanceSplines(self):

        """ Use bivariate splines to interpolate the Z,X grids. Note that I've extrapolated the grids at the point of creation so they spanning
        somewhat wider parameter values than they would have by default."""
        # create list of model coordinates (these are for the pre-computed extrapolated grids)

        self.met_vals = np.array([-1.5,-1.1,-0.5,0.0,0.2,0.5,1.0])

        self.na_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6,0.9,1.2])
        self.mg_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6])
        self.ti_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6])
        self.ca_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6])
        self.fe_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6])
        self.k_vals   = np.array([-0.3,0.0,0.3,0.6])
        self.cbn_vals = np.array([-0.3,-0.15,0.0,0.15,0.3])
        self.teffs    = np.array([-100,-50,0,+50,+100])
        self.ones_vals= np.array([-0.3,0.0,0.3,0.6])


        self.na_surfs = []
        self.mg_surfs = []
        self.ti_surfs = []
        self.ca_surfs = []
        self.fe_surfs = []
        self.k_surfs  = []
        self.cbn_surfs= []
        self.T_surfs= []
        self.ones_surfs = []

        for xx in range(self.nind):

            self.na_surfs.append( rbsplin(self.met_vals,self.na_vals,self.na_grid[:,:,xx]) )
            self.mg_surfs.append( rbsplin(self.met_vals,self.mg_vals,self.mg_grid[:,:,xx],ky=2)) # now the grids are extended,
            self.ti_surfs.append( rbsplin(self.met_vals,self.ti_vals,self.ti_grid[:,:,xx],ky=2)) # is the order still best restricted?
            self.ca_surfs.append( rbsplin(self.met_vals,self.ca_vals,self.ca_grid[:,:,xx],ky=2))
            self.fe_surfs.append( rbsplin(self.met_vals,self.fe_vals,self.fe_grid[:,:,xx],ky=2))
            self.k_surfs.append(  rbsplin(self.met_vals,self.k_vals, self.k_grid[:,:,xx] ,ky=1))
            self.cbn_surfs.append(rbsplin(self.met_vals,self.cbn_vals,self.cbn_grid[:,:,xx],ky=2))
            self.T_surfs.append(  rbsplin(self.met_vals,self.teffs,self.teff_grid[:,:,xx],ky=2))
            self.ones_surfs.append( rbsplin(self.met_vals,self.ones_vals,self.ones_grid[:,:,xx],ky=1))

    def BivarAbundanceLingrid(self):

        """ This is like the BivarAbundanceSplines method, except it uses a linear interpolator instead. In fact, it doesn't make that much difference...
        but the other method keeps the interpolation smooth, whereas this one avoids issues with running into grid edges (i.e. extrapolation)."""

        # create list of model coordinates (these are for the pre-computed extrapolated grids)

        self.met_vals = np.array([-1.5,-1.1,-0.5,0.0,0.2,0.5,1.0])

        self.na_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6,0.9,1.2])
        self.mg_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6])
        self.ti_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6])
        self.ca_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6])
        self.fe_vals  = np.array([-0.6,-0.3,0.0,0.3,0.6])
        self.k_vals   = np.array([-0.3,0.0,0.3,0.6])
        self.cbn_vals = np.array([-0.3,-0.15,0.0,0.15,0.3])
        self.teffs    = np.array([-100,-50,0,+50,+100])
        self.ones_vals= np.array([-0.3,0.0,0.3,0.6])


        self.na_surfs = []
        self.mg_surfs = []
        self.ti_surfs = []
        self.ca_surfs = []
        self.fe_surfs = []
        self.k_surfs  = []
        self.cbn_surfs= []
        self.T_surfs= []
        self.ones_surfs = []

        for xx in range(self.nind):

            self.na_surfs.append( rg_int( (self.met_vals,self.na_vals), self.na_grid[:,:,xx], bounds_error=False,fill_value=None) )
            self.mg_surfs.append( rg_int( (self.met_vals,self.mg_vals), self.mg_grid[:,:,xx], bounds_error=False,fill_value=None) )
            self.ti_surfs.append( rg_int( (self.met_vals,self.ti_vals), self.ti_grid[:,:,xx], bounds_error=False,fill_value=None) )
            self.ca_surfs.append( rg_int( (self.met_vals,self.ca_vals), self.ca_grid[:,:,xx], bounds_error=False,fill_value=None) )
            self.fe_surfs.append( rg_int( (self.met_vals,self.fe_vals), self.fe_grid[:,:,xx], bounds_error=False,fill_value=None) )
            self.k_surfs.append(  rg_int( (self.met_vals,self.k_vals),  self.k_grid[:,:,xx],  bounds_error=False,fill_value=None) )
            self.cbn_surfs.append(rg_int( (self.met_vals,self.cbn_vals),self.cbn_grid[:,:,xx],bounds_error=False,fill_value=None) )
            self.T_surfs.append(  rg_int( (self.met_vals,self.teffs),   self.teff_grid[:,:,xx],      bounds_error=False,fill_value=None) )
            self.ones_surfs.append( rg_int( (self.met_vals,self.ones_vals), self.ones_grid[:,:,xx], bounds_error=False,fill_value=None) )



    def BAS_test(self):

        """ Method introduced to test whether the model works as intended."""
        orig_Z_vals  = self.met_vals#[:5]
        orig_na_vals = self.na_vals#[1:-1]
        orig_na_grid = self.na_grid#[:5,1:-1,:]

        f,axr=plt.subplots(4,5)
        f.set_tight_layout(True)

        for xx in range(self.nind):
            
            ax = axr.flat[xx]

            ax.set_title(self.index_names[xx],fontsize=22)

            for ii in range(orig_Z_vals.size): 

                ax.plot(orig_na_vals,orig_na_grid[ii,:,xx],'o')

                recovered_vals = self.na_surfs[xx](np.ones(orig_Z_vals.size)*orig_Z_vals[ii],orig_na_vals,grid=False)

                ax.plot(orig_na_vals,recovered_vals,'-')




### INITIALISATION SEQUENCE

mmm = modelset()
mmm.GordianSolution()
mmm.BivarAbundanceGrid()
mmm.BivarAbundanceSplines()
#mmm.BivarAbundanceLingrid()

### FUNCTION DEFINITIONS

def predictor(par_mask, theta):
    
    """
    NAME: predictor

    PURPOSE: given a set of stellar population parameters, produces predicted equivalent 
             widths for a set of 20 spectral indices.

    INPUTS: 

    par_mask -- boolean array of length thirteen that determines which parameters will be read in (True) 
                and which will revert to default values defined herein (False).

    theta    -- parameter array. In order:
             
                l_age, X1, X2, Z, p_mg, p_ca, p_na, p_k, p_cbn, p_fe, p_ti, T_eff, p_ONeS
             
                False values in par_mask lead to the removal of the corresponding item in this array. 
             
                e.g. if three parameters are masked, predictor() will expect an array of length 7 containing 
                the non-default values of the parameters which have *not* been masked out.

    OUTPUTS:

    predictions -- array of 20 predicted index strengths

    NOTES: 

    default age is 13.5 Gyr
    default X1 is 1.3
    default X2 is: =X1 if X1 is not masked, 2.3 if X1 is masked

    This means: 
    
    -if X1 and X2 are on, both are fitted
    -if X1 but not X2 is on, fit a single power law IMF
    -if X2 but not X1 is on, fit a Vazdekis-like IMF
    -if neither X1 nor X2 are on, apply a Kroupa-like IMF

    All other defaults(i.e. abundances) are zero.

    """
    
    input_pars = np.zeros(13)    # will contain default parameter values

    # ASSIGN DEFAULT VALUES:

    input_pars[0] = np.log(13.5)  # log(age /Gyr)
    input_pars[1] = 1.3           # IMF slope below 0.5M_solar (X1)
    input_pars[2] = input_pars[1] # IMF slope above 0.5M_solar (X2)

    if (not par_mask[1])&(not par_mask[2]): input_pars[2] = 2.3

    # (the rest are 0. by default)

    # ASSIGN (NON-DEFAULT) PARAMETERS:
    input_pars[np.array(par_mask)] = theta

    # UNPACK THE PARAMETER VALUES:
    l_age, X1, X2, Z, p_mg, p_ca, p_na, p_k, p_cbn, p_fe, p_ti, T_eff, p_ONeS = input_pars

    ### CREATE PREDICTED VALUES FROM GRID:  
    predictions = []
    for ii in range(mmm.nind): predictions.append( mmm.lingridfitter[ii]((X1,X2,l_age,Z)) )
    predictions = np.array(predictions)

    ### LINEAR ADDITION OF ABUNDANCE VARIATIONS:
    na_delt  = np.array([float(fnct(Z,p_na,grid=False)  ) for fnct in mmm.na_surfs  ])
    mg_delt  = np.array([float(fnct(Z,p_mg,grid=False)  ) for fnct in mmm.mg_surfs  ])
    ti_delt  = np.array([float(fnct(Z,p_ti,grid=False)  ) for fnct in mmm.ti_surfs  ])
    ca_delt  = np.array([float(fnct(Z,p_ca,grid=False)  ) for fnct in mmm.ca_surfs  ])
    fe_delt  = np.array([float(fnct(Z,p_fe,grid=False)  ) for fnct in mmm.fe_surfs  ])
    k_delt   = np.array([float(fnct(Z,p_k,grid=False)   ) for fnct in mmm.k_surfs   ])
    cbn_delt = np.array([float(fnct(Z,p_cbn,grid=False) ) for fnct in mmm.cbn_surfs ])
    T_delt   = np.array([float(fnct(Z,T_eff,grid=False) ) for fnct in mmm.T_surfs   ])
    ONeS_delt= np.array([float(fnct(Z,p_ONeS,grid=False)) for fnct in mmm.ones_surfs])
   
    #na_delt  = np.array([float(fnct((Z,p_na))  ) for fnct in mmm.na_surfs  ])  ### if you were to use the linear interpolation method instead
    #mg_delt  = np.array([float(fnct((Z,p_mg))  ) for fnct in mmm.mg_surfs  ])  ### of the bivariate splines over Z,X you would need to uncomment this block
    #ti_delt  = np.array([float(fnct((Z,p_ti))  ) for fnct in mmm.ti_surfs  ])  ### and comment out the one above.
    #ca_delt  = np.array([float(fnct((Z,p_ca))  ) for fnct in mmm.ca_surfs  ])
    #fe_delt  = np.array([float(fnct((Z,p_fe))  ) for fnct in mmm.fe_surfs  ])
    #k_delt   = np.array([float(fnct((Z,p_k))   ) for fnct in mmm.k_surfs   ])
    #cbn_delt = np.array([float(fnct((Z,p_cbn)) ) for fnct in mmm.cbn_surfs ])
    #T_delt   = np.array([float(fnct((Z,T_eff)) ) for fnct in mmm.T_surfs   ])
    #ONeS_delt= np.array([float(fnct((Z,p_ONeS))) for fnct in mmm.ones_surfs])
 
    ### ADD ALL ABUNDANCE-DRIVEN DEVIATIONS TO PREDICTIONS:
    predictions += na_delt
    predictions += mg_delt
    predictions += ti_delt
    predictions += ca_delt
    predictions += fe_delt
    predictions += k_delt
    predictions += cbn_delt
    predictions += T_delt
    predictions += ONeS_delt
    
    return predictions



def fast_predictor(samples):
    
    """
    NAME: fast_predictor

    PURPOSE: given a many sets of stellar population parameters, quickly produces predicted equivalent 
             widths for a set of 21 spectral indices.

    INPUTS: 

    samples    -- MCMC chain (an array of shape [N_steps,13]) containing parameter draws. In order:
             
                l_age, X1, X2, Z, p_mg, p_ca, p_na, p_k, p_cbn, p_fe, p_ti, T_eff, p_ONeS

    OUTPUTS:

    predictions -- array of 21 predicted index strengths x N_steps

    """
    
    input_pars = np.zeros(13)    # will contain default parameter values

    # UNPACK THE PARAMETER VALUES:
    #l_age, X1, X2, Z, p_mg, p_ca, p_na, p_k, p_cbn, p_fe, p_ti, T_eff, p_ONeS = input_pars

    ### CREATE PREDICTED VALUES FROM GRID:  
    predictions = np.zeros( (samples.shape[0],mmm.nind),float )
    for ii in range(mmm.nind): predictions[:,ii] = mmm.lingridfitter[ii]((samples[:,1],samples[:,2],samples[:,0],samples[:,3]))

    ### LINEAR ADDITION OF ABUNDANCE VARIATIONS:
    na_delt  = np.zeros( (samples.shape[0],mmm.nind),float )
    mg_delt  = np.zeros( (samples.shape[0],mmm.nind),float )
    ti_delt  = np.zeros( (samples.shape[0],mmm.nind),float )
    ca_delt  = np.zeros( (samples.shape[0],mmm.nind),float )
    fe_delt  = np.zeros( (samples.shape[0],mmm.nind),float )
    k_delt   = np.zeros( (samples.shape[0],mmm.nind),float )
    cbn_delt = np.zeros( (samples.shape[0],mmm.nind),float )
    T_delt   = np.zeros( (samples.shape[0],mmm.nind),float )
    ONeS_delt= np.zeros( (samples.shape[0],mmm.nind),float )
 
    for ii in range(mmm.nind): 

        na_delt[:,ii]  = mmm.na_surfs[ii](samples[:,3],samples[:,6], grid=False)
        mg_delt[:,ii]  = mmm.mg_surfs[ii](samples[:,3],samples[:,4], grid=False)
        ti_delt[:,ii]  = mmm.ti_surfs[ii](samples[:,3],samples[:,10],grid=False)
        ca_delt[:,ii]  = mmm.ca_surfs[ii](samples[:,3],samples[:,5], grid=False)
        fe_delt[:,ii]  = mmm.fe_surfs[ii](samples[:,3],samples[:,9], grid=False)
        k_delt[:,ii]   = mmm.k_surfs[ii](samples[:,3],samples[:,7],  grid=False)
        cbn_delt[:,ii] = mmm.cbn_surfs[ii](samples[:,3],samples[:,8],grid=False)
        T_delt[:,ii]   = mmm.T_surfs[ii](samples[:,3],samples[:,11], grid=False)
        ONeS_delt[:,ii]= mmm.ones_surfs[ii](samples[:,3],samples[:,12],grid=False)

    ### ADD ALL ABUNDANCE-DRIVEN DEVIATIONS TO PREDICTIONS:
    predictions += na_delt
    predictions += mg_delt
    predictions += ti_delt
    predictions += ca_delt
    predictions += fe_delt
    predictions += k_delt
    predictions += cbn_delt
    predictions += T_delt
    predictions += ONeS_delt
    
    return predictions



def grid_plotter(axis,idx1,idx2,par1,par2,showimf=False,col0="Blues",col1="Oranges"):
    
    """
    grid_plotter plots a grid of model values onto a given axis

    INPUTS:

    axis

    idx1, idx2 -- indexing of outputs from predictor()

    par1, par2 -- NAMES of parameters to be gridded up: l_age, Z, p_mg, p_ca, p_na, p_k, p_cbn, p_fe, p_ti, T_eff, p_ONeS

    NB. plots three grids, one for MW, one for Salpeter, one for X=3


    """


    p_dict = {'l_age':0, 'Z':3, 'p_mg':4, 'p_ca':5, 'p_na':6, 'p_k':7, 'p_cbn':8, 'p_fe':9, 'p_ti':10, 'T_eff':11, 'p_ONeS':12}

    p_ran  = {'l_age': np.log(np.array([7.,9.,11.,13.5])), 
              'Z': np.array([-.8,-.6,-.4,-.2,0.,.2]), 
              'p_mg': np.array([-.3,0,.3]), 
              'p_ca': np.array([-.3,0,.3]), 
              'p_na': np.array([-.3,0,.3,.6,.9]), 
              'p_k':  np.array([-.3,0,.3]), 
              'p_cbn': np.array([-.15,0,.15]), 
              'p_fe': np.array([-.3,0,.3]), 
              'p_ti': np.array([-.3,0,.3]), 
              'T_eff': np.array([-50,0,50]), 
              'p_ONeS': np.array([-.3,0,.3])}

    pvals1 = np.zeros(13)
    pvals2 = np.zeros(13)
    pvals3 = np.zeros(13)

    pvals1[0] = np.log(13.5) # all olds
    pvals2[0] = np.log(13.5)
    pvals3[0] = np.log(13.5)
    
    pvals1[1] = 1.3 ; pvals1[2] = 2.3 # three IMFS
    pvals2[1] = 2.3 ; pvals2[2] = 2.3
    pvals3[1] = 3.0 ; pvals3[2] = 3.0

    pvals1[4] = 0.5 # HACK GRIDS
    #pvals1[6] = 0.7
    #pvals1[9] = -0.1
    pvals1[10]= 0.5
    pvals1[12]=0.5


    # everything else starts as default
    
    param_vals = (pvals1,pvals2,pvals3)

    pidx1 = p_dict[par1]
    pidx2 = p_dict[par2]

    pran1 = p_ran[par1]
    pran2 = p_ran[par2]

    outgrid = np.zeros( (pran1.size,pran2.size,2),float ) # this many predictions: 2 features, pran1 x pran2 predictions

    for imf in range(3):

        pvals = param_vals[imf]
      
        for ii,p1 in enumerate(pran1):
            for jj,p2 in enumerate(pran2):
                
                pvals[pidx1] = p1
                pvals[pidx2] = p2
                
                outgrid[ii,jj,:] = predictor(np.ones(pvals.size,bool), pvals)[[idx1,idx2]]
        
        xvals = outgrid[:,:,0] # two grids of values spanning the two parameter ranges
        yvals = outgrid[:,:,1] # , given some IMF
        
        if imf==0: gridgrad(xvals,yvals,axis,c0=col0,c1=col1)
        if (imf==1)&showimf: gridgrad(xvals,yvals,axis,c0=col0,c1=col1,alpha=0.3)