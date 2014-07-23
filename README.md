## RADD: Race Against Drift-Diffusion model of sensorimotor inhibition and decision-making

## Summary

RADD is a python module for modeling the underlying dynamics of motor inhibition
as a combination of two widely utilized conceptual frameworks: race models of response inhibtion
and drift-diffusion models of decision-making.

![Image of RADD](https://www.evernote.com/shard/s430/sh/8ce6464a-a304-411f-b26c-32162ceba3bc/56d47bbfbaf44a531a03c03fd4a21438/res/cc0f6f25-969a-44e7-a490-eec2386ee6ac/a.ssRe%20Final%20Fits%20and%20Figures.ipynb.jpg?resizeSmall&width=832&alpha=)

RADD seeks to explain both proactive and reactive forms of response inhibition within a unified
framework in which the competition between direct ("Go") and indirect ("No Go") pathways is modeled
as a stochastic accumulation of evidence between "Respond" and "Inhibit" boundaries. This diffusion 
process acts as a dynamically moving baseline from which a hyperdirect "Stop" process can be initiated. 
In the event that a stop signal is encountered, the hyperdirect pathway must override the current 
level of "Go" evidence in order to suppress the evolving motor response.



## Features

* Includes models of proactive and reactive stopping.

* Flexible control over numerous parameters including all standard and inter-trial
  variability parameters of the full drift-diffusion model and additional stop-signal
  parameters.

* Optional temporally dependent dynamic bias signal (see [Hanks et al., 2011](http://www.jneurosci.org/content/31/17/6339.full.pdf))

* Quality visualizations for assessing response time distributions of Go and Stop processes,
  comparing empirical and simulated data, etc.

* Simulate neural integration of direct, indirect, and hyperdirect pathways in the 
  Basal Ganglia - useful for generating and testing predictions about fMRI and 
  single-unit electrophysiological data.



## Future Development

* Currently RADD is in *very early* stages of development and requires the user to define values
  for model parameters (undefined parameters assume default values (which can be found in 
  [Matzke & Wagenmakers, 2009](http://www.ejwagenmakers.com/2009/MatzkeWagenmakers2009.pdf)).
  However, future releases will include parameter fitting routines.

* We are currently in the process of implementing RADD as a neural network model of the BG.
  Ultimately, we aim to merge the two models into a single package for simulating sensorimotor 
  inhibition and decision making across multiple levels, from descriptive cognitive processes 
  to their neurobiological implementation.

* Other goals: easy package management and installation (i.e. via pip or conda), proper 
  documentation and tutorials, goodness of fit statistics and additional model comparison
  methods, +



## Examples

Below is an example of how to simulate several conditions in a typical proactive stop-signal task.
Numerous other "pre-release" (and poorly documented) examples are availabe in the form of 
iPython Notebooks at [RADD IPyNb's](http://nbviewer.ipython.org/github/dunovank/pynb/tree/master/).


##### import libraries & define global parameters
```python
from radd import ss, psy, simfx

a=.37; z=.5*a; Ter=.347; eta=.14; st=.0001; sz=.0001; s2=.01; xpo=[12, 12.29]; pSSD=.450;
mu_ss=-2.17; ssTer=.099; ssRe_TB=.653; ssPro_TB=.6; nt=1000; sTB=.00001; ssTer_var=.0001
```

##### simulate behavior under different probabilities of "Go" as a change in the drift-rate
```python
out=[]

pGo=[.2, .4, .6, .8]
vlist=[0.20, 0.45, 0.60, 0.95]

for i, v in enumerate(vlist):
    
    gp={'a':a, 'z':z, 'v':v, 'Ter':Ter, 'eta':eta, 'st':st, 'sz':sz, 's2':s2}
    sp={'mu_ss':mu_ss, 'pGo':pGo[i], 'ssd':pSSD, 'ssTer':ssTer, 'ssTer_var':ssTer_var}
    
    sim_data=ss.set_model(gParams=gp, sParams=sp, ntrials=nt, timebound=ssPro_TB, 
    	t_exp=True, exp_scale=xpo, visual=False, task='ssProBSL')
    
    out.append(sim_data)
```
