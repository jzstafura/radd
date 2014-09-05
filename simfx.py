#!/usr/local/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
import time
from scipy import stats

def sim_radd(mu, s2, TR, a, z, mu_ss=-1.6, ssd=.450, timebound=0.653, ss_trial=False, exp_scale=[10,10], integrate=False, visual=False, depHyper=True, **kwargs):

	"""

	Standard radd simulation model, 
	including exponential temporal bias 
	and optional BOLD predictions

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
 	tau=.0001			# time per step of the diffusions
	dx=np.sqrt(s2*tau)  # dx is the step size up or down.
	e=z		     		# starting point
	e_ss=z				#arbitrary (positive) init value
	ss_started=False
	elist=[]; tlist=[]; elist_ss=[]; tlist_ss=[]; ithalamus=[];
	num=exp_scale[0]
	denom=exp_scale[1]
	thalamus=z
	ss_ti=0
	
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
		
			# This is the probability of moving up or down from z.
			# If mu is greater than 0, the diffusion tends to move up.
			# If mu is less than 0, the diffusion tends to move down.
			p=0.5*(1 + mu*dx/s2)
			
			# if r < p then move up
			if r < p:
				e = e + dx# + timebias	
				
			# else move down
			else:
				e = e - dx# + timebias
			
			elist.append(e)
			tlist.append(t)
			
		if ss_trial and t>=ssd:
			
			r_ss=np.random.random_sample()
			p_ss=0.5*(1 + mu_ss*dx/s2)
			
			#test if stop signal has started yet.
			#if not, then start at current position of "go/nogo" DV: e
			if not ss_started and depHyper:
				ss_started=True
				e_ss=e
				
			else:
				# if r < p then move up
				if r_ss < p_ss:
					e_ss = e_ss + dx
					ss_ti = dx
				# else move down
				else:
					e_ss = e_ss - dx
					ss_ti = -dx
			
			elist_ss.append(e_ss)
			tlist_ss.append(t)
		
		if integrate:

			#if e!=z and e_ss!=z:
			if len(elist)>0 and len(elist_ss)>0:
				thalamus = e + ss_ti

			elif len(elist)>0 and len(elist_ss)==0:				
				thalamus = e

			else:
				thalamus = e_ss

			ithalamus.append(thalamus)
	

	evidence_lists=[elist, elist_ss]
	timestep_lists=[tlist, tlist_ss]
	
	if choice is None:
		if e >= a:
			choice = 'go'
		elif e<=0 or e_ss<=0:
			choice = 'stop'
	
	if not integrate:
		ithalamus='null'
	
	if visual:
		return t, choice, evidence_lists, timestep_lists, ithalamus
	else:
		return t, choice


def thal(mu, s2, TR, a, z, mu_ss=-1.6, ssd=.450, timebound=0.653, ss_trial=False, exp_scale=[12, 12.29], integrate=False, visual=False, depHyper=True, **kwargs):

	"""
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
	tb=0
	tau=.0001		# time per step of the diffusion
	dx=np.sqrt(s2*tau)  	# dx is the step size up or down.
	gti=0; ss_ti=0
	thalamus=[z]
	tsteps=[0]
	thal_i=z
	choice=None
	denom=exp_scale[1]
	num=exp_scale[0]
	no_choice_yet=True

	if TR>ssd and ss_trial:
		t=ssd	# start the time at ssd
	else:		# or
		t=TR	# start the time at TR

	while t<timebound: #e<a and e>0 and e_ss>0: 
		
		t = t + tau
		if t >= TR:
			
			#tb = tb + tau
			#timebias=(np.exp(exp_scale[0]*tb))/(np.exp(exp_scale[1]))
		
			# r is between 0 and 1
			r=np.random.random_sample()
			p=0.5*(1 + mu*dx/s2)
			
			if r < p:
				gti = dx #+ timebias
			else:
				gti = -dx #+ timebias	

		if ss_trial and t >= ssd:
			r_ss=np.random.random_sample()
			p_ss=0.5*(1 + mu_ss*dx/s2)
			
			if r_ss < p_ss:
				ss_ti = dx
			else:
				ss_ti = -dx

		thalamus.append(thalamus[-1]+gti+ss_ti)
		tsteps.append(t)
		
		if thalamus[-1]>=a and no_choice_yet:
			rt=t
			choice = 'go'
			no_choice_yet=False
	
	if choice=='go':#np.array(thalamus).max() >= a:
		paths=[thalamus, [0]]
		timesteps=[tsteps, [0]]
	else:
		choice = 'stop'
		rt = t
		paths = [[0], thalamus]
		timesteps = [[0],tsteps]

	if visual:
		return rt, choice, paths, timesteps, thalamus
	else:
		return rt, choice


def simIndependentPools(mu, s2, TR, a, z, mu_ss=-1.6, ssd=.450, timebound=0.653, ss_trial=False, exp_scale=[12, 12.29], integrate=False, visual=False, depHyper=True, **kwargs):

	tb=0
	tau=.0001		# time per step of the diffusion
	dx=np.sqrt(s2*tau)  	# dx is the step size up or down.
	choice=None
	denom=exp_scale[1]; num=exp_scale[0]
	a=a-z; b=-z; z=0
	gti=0; ss_ti=0
	thalamus=[z]; tsteps=[TR]
	hyper_tsteps=[ssd]; hyperthal=[z]; no_choice_yet=True
	normp=True

	if TR>ssd and ss_trial:
		t=ssd	# start the time at ssd
	else:		# or
		t=TR	# start the time at TR


	while t<timebound: #e<a and e>0 and e_ss>0: 
		
		t = t + tau
		
		if t >= TR:
			
			#tb = tb + tau
			#timebias=(np.exp(exp_scale[0]*tb))/(np.exp(exp_scale[1]))
		
			# r is between 0 and 1
			r=np.random.random_sample()
			p=0.5*(1 + mu*dx/s2)
			
			if r < p:
				gti = dx #+ timebias
			else:
				gti = -dx #+ timebias

			thalamus.append(thalamus[-1]+gti)
			tsteps.append(t)	

		if ss_trial and t >= ssd:

			r_ss=np.random.random_sample()
			p_ss=0.5*(1 + mu_ss*dx/s2)

			if r_ss < p_ss:
				ss_ti = dx
			else:
				ss_ti = -dx

			hyper_tsteps.append(t)
			hyperthal.append(hyperthal[-1]+ss_ti)
		
		if thalamus[-1]+hyperthal[-1]>=a and no_choice_yet:
			rt=t
			mu=0
			choice = 'go'
			no_choice_yet = False

		if thalamus[-1]+hyperthal[-1]<=b and no_choice_yet:
			rt = 0
			mu_ss=0
			choice = 'stop'
			no_choice_yet = False

	if choice is not None:
		paths = [thalamus, hyperthal]
		timesteps = [tsteps, hyper_tsteps]
	elif choice is None and t>=timebound:
		choice='stop'; rt=t
		paths = [thalamus, hyperthal]
		timesteps = [tsteps, hyper_tsteps]
	else:
		print "what the fuck..."

	if visual:
		return rt, choice, paths, timesteps, normp
	else:
		return rt, choice



def sustained_integrator(mu, s2, TR, a, z, mu_ss=-1.6, ssd=.450, timebound=0.653, ss_trial=False, exp_scale=[10,10], integrate=False):

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
	dx=np.sqrt(s2*tau)  # dx is the step size up or down.
	e=z		     		# starting point
	e_ss=z				#arbitrary (positive) init value
	ss_started=False
	elist=[]; tlist=[]; elist_ss=[]; tlist_ss=[]; ithalamus=[];
	num=exp_scale[0]
	denom=exp_scale[1]
	thalamus=z
	ss_ti=0

	# loop until evidence is greater than or equal to a (upper boundary)
	# or evidence is less than or equal to 0 (lower boundary)
	while t<timebound: #e<a and e>0 and e_ss>0: 
			
		#if t>=timebound:
		#	choice='stop'
		#	break
		
		# increment the time
		t = t + tau
		
		if e>=a or e_ss<=0:
			
			ithalamus.append(thalamus)

			continue

		if t>=TR and e<a:
			
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
				gti = dx + timebias
			
			# else move down
			else:
				e = e - dx + timebias
				gti = -dx + timebias
			
			elist.append(e)
			tlist.append(t)
			
		if ss_trial and t>=ssd:
			
			r_ss=np.random.random_sample()
			p_ss=0.5*(1 + mu_ss*dx/s2)
			
			#test if stop signal has started yet.
			#if not, then start at current position of "go/nogo" DV: e
			if not ss_started:
				ss_started=True
				e_ss=e
				ss_ti = -dx
				
			else:
				# if r < p then move up
				if r_ss < p_ss:
					e_ss = e_ss + dx
					ss_ti = dx
				
				# else move down
				else:
					e_ss = e_ss - dx
					ss_ti = -dx
			
			elist_ss.append(e_ss)
			tlist_ss.append(t)

		
		if len(elist)>0 and len(elist_ss)>0:
			thalamus = e + ss_ti

		elif len(elist)>0 and len(elist_ss)==0:				
			thalamus = e

		else:
			thalamus = e_ss

		ithalamus.append(thalamus)


	evidence_lists=[elist, elist_ss]
	timestep_lists=[tlist, tlist_ss]
	
	if choice is None:
		if e >= a:
			choice = 'go'
		elif e<=0 or e_ss<=0:
			choice = 'stop'
		elif e<a and t>=timebound:
			choice = 'stop'

	return t, choice, evidence_lists, timestep_lists, ithalamus


def integrator(mu, s2, TR, a, z, mu_ss=-1.6, ssd=.450, timebound=0.653, ss_trial=False, exp_scale=[10,10]):

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
	
	tb=0					# init the exp time bias to 0
	choice=None				# init choice as NoneType
 	tau=.0001				# time per step of the diffusion
	dx=np.sqrt(s2*tau)  	# dx is the step size up or down.
	e=z		     			# init go/nogo at starting point
	e_ss=z					# init ss at starting point  
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
				gti = dx + timebias
			
			# else move down
			else:
				e = e - dx + timebias
				gti = -dx + timebias
			
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
				ss_ti = -dx
				
			else:
				# if r < p then move up
				if r_ss < p_ss:
					e_ss = e_ss + dx
					ss_ti = dx
				
				# else move down
				else:
					e_ss = e_ss - dx
					ss_ti = -dx
			
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

	return t, choice, evidence_lists, timestep_lists, ithalamus


def sim_ss(mu, s2, TR, a, z, mu_ss=-6, ssd=.450, timebound=0.653, ss_trial=False, **kwargs):

	"""
	
	Simple radd simulation model (no exponential time bias)
	
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
	
	else:		
	
		t=TR	# start the time at TR

 	tau=.0001			# time per step of the diffusion
	choice=None			# init choice as NoneType
	dx=np.sqrt(s2*tau)  # calculate dx (step size)
	e=z		        # starting point
	e_ss=z	  		# arbitrary (positive) init value
	ss_started=False
	elist=[]; tlist=[]; elist_ss=[]; tlist_ss=[]

	# loop until evidence is greater than or equal to a (upper boundary)
	# or evidence is less than or equal to 0 (lower boundary)
	while e<a and e>0 and e_ss>0:

		if t>=timebound:
			choice='stop'
			break

		# increment the time
		t = t + tau

		# random float between 0 and 1
		r=np.random.random_sample()

		# This is the PROBABILITY of moving up or down.
		# If mu is greater than 0, the diffusion tends to move up.
		# If mu is less than 0, the diffusion tends to move down.
		p=0.5*(1 + mu*dx/s2)

		# if r < p then move up
		if r < p:
			e = e + dx
		# else move down
		else:
			e = e - dx

		if ss_trial and t>=ssd:

			r_ss=np.random.random_sample()
			p_ss=0.5*(1 + mu_ss*dx/s2)
			
			#test if stop signal has started yet.
			#if not, then start at current position of "go/nogo" DV: e
			if not ss_started:
				ss_started=True
				e_ss=e

			else:
				# if r < p then move up
				if r_ss < p_ss:
					e_ss = e_ss + dx
				# else move down
				else:
					e_ss = e_ss - dx

			elist_ss.append(e_ss)
			tlist_ss.append(t)

		elist.append(e)
		tlist.append(t)

	evidence_lists=[elist, elist_ss]
	timestep_lists=[tlist, tlist_ss]
	
	if choice is None:
		if e >= a:
			choice = 'go'
		elif e<=0 or e_ss<=0:
			choice = 'stop'

	return t, choice, evidence_lists, timestep_lists, 1.0

def sim_ddm(mu, s2, TR, a, z, mu_ss=-1.6, ssd=.450, timebound=0.653, ss_trial=False, exp_scale=[10,10], integrate=False):

	"""

	Standard radd simulation model, 
	including exponential temporal bias 
	and optional BOLD predictions

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
	dx=np.sqrt(s2*tau)  # dx is the step size up or down.
	e=z		     		# starting point
	e_ss=z				#arbitrary (positive) init value
	ss_started=False
	elist=[]; tlist=[]; elist_ss=[]; tlist_ss=[]; ithalamus=[];
	num=exp_scale[0]
	denom=exp_scale[1]
	thalamus=z
	ss_ti=0
	
	# loop until evidence is greater than or equal to a (upper boundary)
	# or evidence is less than or equal to 0 (lower boundary)
	while e<a and e>0 and e_ss>0: 
			
		if t>=timebound:
			choice='stop'
			break
		
		# increment the time
		t = t + tau
		
		if t>=TR:
			

			tb = tb + tau
			timebias=(np.exp(num*tb))/(np.exp(denom))
		
			# r is between 0 and 1
			r=np.random.random_sample()
		
			# This is the probability of moving up or down from z.
			# If mu is greater than 0, the diffusion tends to move up.
			# If mu is less than 0, the diffusion tends to move down.
			p=0.5*(1 + mu*dx/s2)
			
			# if r < p then move up
			if r < p:
				e = e + dx + timebias	
				
			# else move down
			else:
				e = e - dx + timebias
			
			elist.append(e)
			tlist.append(t)
			
		if ss_trial and t>=ssd:
			
			r_ss=np.random.random_sample()
			p_ss=0.5*(1 + mu_ss*dx/s2)
			
			#test if stop signal has started yet.
			#if not, then start at current position of "go/nogo" DV: e
			if not ss_started:
				ss_started=True
				e_ss=z
				
			else:
				# if r < p then move up
				if r_ss < p_ss:
					e_ss = e_ss + dx
					ss_ti = dx
				# else move down
				else:
					e_ss = e_ss - dx
					ss_ti = -dx
			
			elist_ss.append(e_ss)
			tlist_ss.append(t)
		
		if integrate:

			#if e!=z and e_ss!=z:
			if len(elist)>0 and len(elist_ss)>0:
				thalamus = e + ss_ti

			elif len(elist)>0 and len(elist_ss)==0:				
				thalamus = e

			else:
				thalamus = e_ss

			ithalamus.append(thalamus)
	

	evidence_lists=[elist, elist_ss]
	timestep_lists=[tlist, tlist_ss]
	
	if choice is None:
		if e >= a:
			choice = 'go'
		elif e<=0 or e_ss<=0:
			choice = 'stop'
	
	if not integrate:
		ithalamus='null'
	return t, choice, evidence_lists, timestep_lists, ithalamus


def sim_ssex(mu, s2, TR, a, z, mu_ss=-1.6, ssd=.450, timebound=0.653, ss_trial=False, exp_scale=[10,10]):


	#DEPRECATED METHOD
	

	"""
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
	
	#if TR>ssd:
	#	t=ssd	# start the time at ssd
	#else:		# or
	#	t=TR	 # start the time at TR
	
	t=TR		        	# start the time at TR
	tb=0				# init the exp time bias to 0
	choice=None			# init choice as NoneType
 	tau=.0001			# time per step of the diffusion
	dx=np.sqrt(s2*tau)  		# dx is the step size up or down.
	e=z;		     		# starting point
	e_ss=10000			#arbitrary (positive) init value
	ss_started=False
	elist=[]; tlist=[]; elist_ss=[]; tlist_ss=[]
	num=exp_scale[0]
	denom=exp_scale[1]
	
	# loop until evidence is greater than or equal to a (upper boundary)
	# or evidence is less than or equal to 0 (lower boundary)
	while e<a and e>0 and e_ss>0: 
			
		if t>=timebound:
			choice='stop'
			break
		
		# increment the time
		t = t + tau
		tb = tb + tau
		timebias=(np.exp(num*tb))/(np.exp(denom))
		
		# r is between 0 and 1
		r=np.random.random_sample()
	
		# This is the PROBABILITY of moving up or down.
		# If mu is greater than 0, the diffusion tends to move up.
		# If mu is less than 0, the diffusion tends to move down.
		p=0.5*(1 + mu*dx/s2)
		p_ss=0.5*(1 + mu_ss*dx/s2)
		
		#if r < p then move up
		if r < p:
			e = e + dx + timebias	
		# else move down
		else:
			e = e - dx + timebias
			
		if ss_trial and t>=ssd:
			
			#test if stop signal has started yet.
			#if not, then start at current position of "go/nogo" DV: e
			if not ss_started:
				ss_started=True
				e_ss=e
				
			else:
				# if r < p then move up
				if r < p_ss:
					e_ss = e_ss + dx
				# else move down
				else:
					e_ss = e_ss - dx
					
			elist_ss.append(e_ss)
			tlist_ss.append(t)
		
		elist.append(e)
		tlist.append(t)
	
	evidence_lists=[elist, elist_ss]
	timestep_lists=[tlist, tlist_ss]
	
	if choice is None:
		if e >= a:
			choice = 'go'
		elif e<=0 or e_ss<=0:
			choice = 'stop'
			
	return t, choice, evidence_lists, timestep_lists



