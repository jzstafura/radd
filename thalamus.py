#!/usr/local/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
import time
from scipy import stats



def integrator(mu, s2, TR, a, z, mu_ss=-1.6, ssd=.450, timebound=0.653, ss_trial=False, time_bias=0, exp_scale=[10,10], integrate=False):

	"""

	alternative, experimental method for integrating go/nogo 
	and ss inputs to the thalamus (default method is ssex.sim_exp(... integrate=True))

	args:
		:: mu = mean drift-rate
		:: s2 = diffusion coeff
		:: TR = non-decision time
		:: a  = boundary height
		:: z  = starting-point 

	returns:
	 	rt (float): 	decision time
		choice (str):	a/b 		
		elist (list):	list of sequential/cumulative evidence
		tlist (list):	list of sequential timesteps
	"""
	
	if TR>ssd and ss_trial:
		t=ssd	# start the time at ssd

	else:		# or
		t=TR	 # start the time at TR
	
	tb=0				# init the exp time bias to 0
	choice=None			# init choice as NoneType
 	tau=.0001			# time per step of the diffusion
	dx=np.sqrt(s2*tau)  		# dx is the step size up or down.
	e=z		     		# starting point
	e_ss=z				#arbitrary (positive) init value
	ss_started=False
	elist=[]; tlist=[]; elist_ss=[]; tlist_ss=[]; ithalamus=[];
	num=exp_scale[0]
	denom=exp_scale[1]
	thalamus=z
	ti=0; gti=0; ss_ti=0
	
	# loop until evidence is greater than or equal to a (upper boundary)
	# or evidence is less than or equal to 0 (lower boundary)
	while e<a and e>0 and e_ss>0: 
			
		if t>=timebound:
			choice='stop'
			break
		
		# increment the time
		t = t + tau
		
		if t>=TR:
			
			#if tb==0:
				#timebias=0
			#else:
			tb = tb + tau
			timebias=(np.exp(num*tb))/(np.exp(denom))
		
			# r is between 0 and 1
			r=np.random.random_sample()
		
			# This is the PROBABILITY of moving up or down.
			# If mu is greater than 0, the diffusion tends to move up.
			# If mu is less than 0, the diffusion tends to move down.
			p=0.5*(1 + mu*dx/s2)
			
			# if r < p then move up
			if r < p:
				e = e + dx + timebias	
				gti= dx + timebias
			
			# else move down
			else:
				e = e - dx + timebias
				gti= -dx + timebias
			
			elist.append(e)
			tlist.append(t)
			
		if ss_trial and t>=ssd:
			
			r_ss=np.random.random_sample()
			p_ss=0.5*(1 + mu_ss*dx/s2)
			#ti=-dx
			#test if stop signal has started yet.
			#if not, then start at current position of "go/nogo" DV: e
			if not ss_started:
				ss_started=True
				e_ss=e
				ss_ti=-dx
				
			else:
				# if r < p then move up
				if r_ss < p_ss:
					e_ss = e_ss + dx
					ss_ti=dx
				
				# else move down
				else:
					e_ss = e_ss - dx
					ss_ti=-dx
			
			elist_ss.append(e_ss)
			tlist_ss.append(t)
		
		if integrate:

			ti = ti + gti + ss_ti

			ithalamus.append(ti)
	

	evidence_lists=[elist, elist_ss]
	timestep_lists=[tlist, tlist_ss]
	
	if choice is None:
		if e >= a:
			choice = 'go'
		elif e<=0 or e_ss<=0:
			choice = 'stop'
	if integrate:
		return t, choice, evidence_lists, timestep_lists, ithalamus
	else:
		return t, choice, evidence_lists, timestep_lists
