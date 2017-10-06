---------------------
  SPINDRIFT README
---------------------

This is SPINDRIFT, [S]tellar [P]opulation [IN]ference [DR]iven by [I]ndex [F]itting [T]echniques.


BACKGROUND:

Unresolved stellar populations (e.g. the stars in other galaxies) may look fuzzy in images,
but the information from the individual stars that comprise the population is still in there
somewhere, waiting to be extracted by the enterprising researcher!

Spectroscopic information can been used for this purpose. If we look at the absorption lines
that are imprinted on the spectrum of a given star, the strength of those features is down to 
detailed atmospheric physics. Historically, such lines were used to classify stars into different 
spectral types corresponding to stars with different physical properties (mass, age, chemical
composition etc.). Taken in aggregate, the strength of certain features in a galaxy spectrum 
-- which is just the sum of the light from all that stars in the galaxy -- offers clues about 
how dominant stars of a particular type are in that galaxy.


METHOD:

SPINDRIFT assumes that you have a set of measurements and appropriate statistical uncertainties
 of line strengths in a galaxy spectrum, and that you wish to fit a model to this data. The model 
will need a few parameters that define things such as the age, chemical composition, and 
distribution of initial masses of stars in the population (and we assume that these are the same 
for all stars in the population, even though they probably aren't -- but see my Thesis for a 
discussion of this and other limitations). 

SPINDRIFT_index_model takes a set of line strength measurements corresponding to a grid of 
varying underlying parameters that is provided by the user. It generates a model object that 
consists of interpolations over this grid and provides a function that can repeatedly invoke 
this model object with different set of parameter values, thereby *predicting* an appropriate
set of line strength values for that choice of parameters.

SPINDRIFT_MCMC makes use of this to evaluate a posterior probability distribution over a chosen 
set of parameters for the user provided data and statistical uncertainties. It does this using 
a Monte-Carlo Markov-Chain method. For this purpose it makes use of the emcee (MCMC ensemble 
sampler) module, and for plotting the results uses the corner module (both of these created by 
D. Foreman-Mackey, github.com/dfm). It allows the user to select the number of MCMC walkers,
steps-per-walker, and burn-in steps (number to be discarded) and can automatically save the 
output as you go along.


NOTES:

SPINDRIFT_example outlines a quick test-case to illustrate how to use the scripts described above.

Additional details of the code and the background to the science can be found in 
'The spatially-resolved stellar populations of nearby early-type galaxies', the author's Thesis.


ACKNOWLEDGMENTS: 

Thanks are due to my PhD supervisors R. Smith and J. Lucey, and also to C. Conroy for 
providing the full-spectrum models I used to produce my grids of line-strengths.