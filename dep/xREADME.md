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

* Gradient descent optimization of drift-diffusion parameters.

* Flexible control over parameter dependencies.

* Include dynamic bias signal (see [Hanks et al., 2011](http://www.jneurosci.org/content/31/17/6339.full.pdf))

* Visualizations for assessing go and stop RT distributions,
  comparing alternative model fits, overlaying simulated data on empirical means, etc.

* Simulate neural integration of direct, indirect, and hyperdirect pathways in the 
  Basal Ganglia - useful for generating and testing predictions about fMRI and 
  single-unit electrophysiological data.