## RADD: Race Against Drift-Diffusion model of sensorimotor inhibition and decision-making

## Summary

RADD is a python package for modeling the underlying dynamics of motor inhibition
as a combination of two widely utilized conceptual frameworks: race models of response inhibtion
and drift-diffusion models of decision-making.

RADD seeks to explain both proactive and reactive forms of response inhibition within a unified
framework in which the competition between direct ("Go") and indirect ("No Go") pathways is modeled
as a stochastic accumulation of evidence between "Respond" and "Inhibit" boundaries. This diffusion 
process acts as a dynamically moving baseline from which a hyperdirect "Stop" process can be initiated. 
In the event that a stop signal is encountered, the hyperdirect pathway must override the current 
level of "Go" evidence in order to suppress the evolving motor response.



## Features

* Includes models of proactive and reactive stopping.

* Flexible control over parameter dependencies.

* Optimize "Go" parameters using wrapper for HDDM ([Wiecki, Sofer, & Frank, 2014](http://journal.frontiersin.org/Journal/10.3389/fninf.2013.00014/abstract))

* Include dynamic bias signal (see [Hanks et al., 2011](http://www.jneurosci.org/content/31/17/6339.full.pdf))

* Visualizations for assessing go and stop RT distributions,
  comparing alternative model fits, overlaying simulated data on empirical means, etc.

* Simulate neural integration of direct, indirect, and hyperdirect pathways in the 
  Basal Ganglia - useful for generating and testing predictions about fMRI and 
  single-unit electrophysiological data.



## Future Goals

* easy package management and installation (i.e. via pip or conda), proper 
  documentation and tutorials, goodness of fit statistics and additional model comparison
  methods, +



## Examples

Below is an example of how to fit and simulate a model. Numerous other "pre-release" (and poorly 
documented) examples are availabe as [IPython Notebooks](http://nbviewer.ipython.org/github/dunovank/pynb/tree/master/SS/).


##### optimize go parameters (via [HDDM](https://github.com/hddm-devs/hddm))
```python
import pandas as pd
from radd import ss, ft, qsim

#load go trials into a pandas df
data=pd.read_csv(SS_AllSx_Data.csv")

#estimate individual subject parameters for go trials
vbias_stats=ft.fit_sxhddm(data, depends_on={'v':'Cond'}, bias=True, informative=True, include=['a', 't', 'v', 'z', 'sv'], 
                   task='ssRe', save_str="vBP")
```

##### simulate range of stop signal drift-rates nested in optimized go parameters
```python

#define range of ss drift rates to simulate
ssvlist = -1*np.arange(.5, .9,.05)
sim_list=[]

#simulate (returns df of simulated trial data)
simdf = qsim.sim_ssv_range(params=vbias_stats, ssvlist=ssvlist, task='ssRe', ntrials=500)
    
```


#### References

Hanks, T. D., Mazurek, M. E., Kiani, R., Hopp, E., & Shadlen, M. N. (2011). Elapsed decision time affects the weighting of prior probability in a perceptual decision task. The Journal of Neuroscience, 31(17), 6339-6352.

Matzke, D., & Wagenmakers, E. J. (2009). Psychological interpretation of the ex-Gaussian and shifted Wald parameters: A diffusion model analysis. Psychonomic Bulletin & Review, 16(5), 798-817.

Wiecki TV, Sofer I and Frank MJ (2013). HDDM: Hierarchical Bayesian estimation of the Drift-Diffusion Model in Python. Front. Neuroinform. 7:14. 