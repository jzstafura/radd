#!usr/bin/env python

from __future__ import division
import os
from collections import OrderedDict
import seaborn as sns
import pandas as pd
from radd import qsim, psy, utils, ft, simfx, ss
from radd.simfx import *
import scipy as scp
from lmfit import minimize, Parameters, Parameter, report_fit, Minimizer, fit_report
import numpy as np
import Scipy.Special

def get_sx_params(sxdf):

	params=sxdf['mean']; params.index=sxdf['param']
	a=params['a']*.1
	z=a*params['z']
	v=params['v(bsl)']*.1

	sp={'ssTer':0.0, 'ssTer_var':0.0, 'pGo':0.0}
	gp={'a':a, 'v':v, 'z':z, 'Ter':params['t'], 'eta':params['sv'], 'st':0.0, 'sz':0.0}

	return [gp, sp]

def raddMinFunc(p, gp, sp, ydata, ntrials=1000):

	ssdlist=np.arange(.20, .45, .05)
	simdf_list=[]
	sp['mu_ss'] = -p['mu_ss'].value
	sp['v'] = p['v'].value
	gp['a'] = p['a'].value
	gp['z'] = p['z'].value
	gp['Ter'] =  p['Ter'].value

	for ssd in ssdlist:

		sp['ssd']=ssd
		out=ss.set_model(gParams=gp, sParams=sp, mfx=simfx.sim_radd, ntrials=ntrials, timebound=.650, 
		                 depHyper=True, visual=False, task='ssRe', return_all_beh=True, condition_str='bsl') 
		
		simdf_list.append(out)

	simdf=pd.concat(simdf_list)
	ym=simdf.groupby('ssd').mean()['acc'].values

	residuals=ym-ydata
	#residuals = scp.sqrt(residuals ** 2 / ysigma ** 2)

	return residuals


def fit_model(data, dep={}, params=[], intervar=False, ntrials=2000, maxiter=100):


	#free=dep.keys
	#cond=dep.values
	#pb=utils.PBinJ(n, color="#4168B7")

	ydata=data.groupby('ssd').mean()['acc'].values[:-1]

	fitdf=pd.DataFrame(columns=['residuals', 'ydata', 'yhat', 'chisq', 'redchi','sse'], 
		index=data.subj_idx.unique())

	p=Parameters()
	p.add('mu_ss', value=1.0, min=0.1, max=2.5)
	p.add('a', value=.5, min=0.1, max=2.5)
	p.add('v', value=1.0, min=0.1, max=2.5)
	p.add('Ter', value=.3, min=0.1, max=2.5)
	p.add('z', value=.35, min=0.01, max=)

	if intervar:

		p.add('eta', value=inits[sx], min=0.1, max=2.5)
		p.add('st', value=inits[sx], min=0.1, max=2.5)
		p.add('sv', value=inits[sx], min=0.1, max=2.5)


	theta = Minimizer(ssvMinFunc, p, fcn_args=(gp,sp,ydata,ntrials), 
	            kws={"full_output":True, "disp":True, "retall":True}) 

		theta.prepare_fit()

		theta.fmin(maxfun=maxiter, ftol=1.e-3, xtol=1.e-3)

	#fitdf.loc[sx]=pd.Series({'ssd':np.arange(.20, .45, .05), "residuals":theta.residual,"ydata":ydata.astype('float'),
	#	'yhat':ydata+theta.residual,"chisq":theta.chisqr,"redchi":theta.redchi, 'sse':np.sum([i**2 for i in theta.residual])})

	return theta
