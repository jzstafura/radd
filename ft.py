#!usr/bin/env python
from __future__ import division
import os, scipy, hddm
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import ss, psy, simfx
from simfx import *
from utils import find_path
from patsy import dmatrix
from lmfit import minimize, Parameters, Parameter, report_fit, Minimizer, fit_report

def make_fit_ssv(gp, sp, task='ssPro'):

	def fit_ssv(x, ssv):
		
		yhat=[]
		for xi in x:
			
			if 'Re' in task:
				sp['ssd']=xi
				sp['pGo']=.5
				output_index=2
			else:
				sp['pGo']=xi
				sp['ssd']=.450
				output_index=1

			sp['mu_ss']=ssv

			sim_data=ss.set_model(gParams=gp, sParams=sp, mfx=thal, ntrials=1000, timebound=.560, task=task)
			
			yhat.append(sim_data[output_ix])

		return yhat

	return fit_ssv


def z_link_func(x, data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return 1 / (1 + np.exp(-(x * stim)))

def v_link_func(x, data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return x * stim

def aic(model):
	k = len(model.get_stochastics())
	logp = sum([x.logp for x in model.get_observeds()['node']])
	return 2 * k - 2 * logp

def bic(model):
	k = len(model.get_stochastics())
	n = len(model.data)
	logp = sum([x.logp for x in model.get_observeds()['node']])
	return -2 * logp + k * np.log(n)

def dic(model):
	return model.dic
	
def fit_sxhddm(data, depends_on={}, include=['a', 't', 'v'], bias=False, informative=False, task='ssRe', save_str='x', **kwargs):
	
	pth=find_path()
	
	grp_dict={}; subj_params=[]; aic_list=[]; bic_list=[]; dic_list=[]

	if bias and 'z' not in include: 
		include.append('z')

	for subj_idx, subj_data in data.groupby('subj_idx'):
		
		try:
			m_sx=hddm.HDDM(subj_data, depends_on=depends_on, bias=bias, 
				include=include, informative=informative)
		
		except Exception:
			continue

		try:
			m_sx.sample(50000, burn=10000, thin=5)
		
		except Exception:
			continue
		
		sx_df=stats_df(m_sx)
		sx_df['subj_idx']=[subj_idx]*len(sx_df)
		
		if subj_idx==data.subj_idx.min():
			m_sx.plot_posteriors(save=True)

		m_sx.print_stats(str(subj_idx)+'_stats.txt')

		subj_params.append(sx_df)
		
		aic_list.append(aic(m_sx)) 
		bic_list.append(bic(m_sx))
		dic_list.append(m_sx.dic)

	allsx_df=pd.concat(subj_params)
	allsx_df.to_csv(save_str+"_SxStats.csv", index=False)
	
	ic_df=pd.DataFrame({'aic':aic_list, 'bic':bic_list, 'dic':dic_list})
	ic_df.to_csv(save_str+"_FitRanks.csv")
	
	return allsx_df


def stats_df(model, save=False):
	"""
	RETURNS: 1
		*model_stats (pandas DataFrame):	same as hddm.HDDM.gen_stats() with 
							column added for parameter names
							(usually call this "fulldf")
	"""
	if not hasattr(model, 'columns'):
		model_stats=model.gen_stats()
		model_stats['param']=model_stats.index
	else:
		model_stats=model

	slist=list()
	for i in model_stats['param']:
		x=i.split('.')
		if x[-1].isdigit():
			sint=int(x[-1])
			slist.append(sint)	
		else: slist.append("GRP")
	model_stats['subj_idx']=slist
	
	if save:
		model_stats.to_csv('fulldf.csv', index=False)
	
	return model_stats

def stats_summary(param_df):
	
	summary_df=pd.pivot_table(param_df, values=['mean', 'std', '2.5q', '25q', '50q', '75q', '97.5q'], index='param', aggfunc=np.average)

	return summary_df

def get_MeanHDDM(df, depends='v', plist=['a', 't', 'v'], zbias=False, inter_var=False):

	#plist.remove(depends)

	if zbias:
		plist.append('z')
	if inter_var:
		plist.extend(['sv', 'st', 'sz'])

	plist.remove(depends)

	summary=stats_summary(df)['mean']
	
	param_dict={}
	for par in plist:
		param_dict[par]=summary.ix[par]

	dep_plist=[summary.ix[p] for p in summary.index if depends in p]
	dep_names=[p.split('(')[1][:-1] for p in summary.index if depends in p]
	
	param_dict[depends+'list']=dep_plist
	param_dict[depends+'conds']=dep_names

	return param_dict


def init_ssfitfx(gp, sp, task='ssPro'):

	if 'Re' in task:
		x=np.arange(.25,.5, .05)
		#y_bsl=
		#y_pnl=

		ydata=np.array([y_bsl, y_pnl])
	else:
		x=np.array([1.0,2.0], dtype='float') #np.arange(0, 1, .2)
		y_hi=np.mean([0.931, 0.744, 0.471])
		y_lo=np.mean([0.240, 0.034, 0.005])
		ydata=np.array([y_hi, y_lo], dtype='float')

	popt, pcov = curve_fit(make_fit_ssv(gp, sp, task=task), x, ydata, sp['mu_ss'])

	return [popt, pcov]


def chisqg(ydata,ymod,sd=None):

      	"""  
	Returns the chi-square error statistic as the sum of squared errors between  
	Ydata(i) and Ymodel(i). If individual standard deviations (array sd) are supplied,   
	then the chi-square error statistic is computed as the sum of squared errors  
	divided by the standard deviations.     Inspired on the IDL procedure linfit.pro.  
	See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.  
	  
	x,y,sd assumed to be Numpy arrays. a,b scalars.  
	Returns the float chisq with the chi-square statistic.  
	  
	Rodrigo Nemmen  
	http://goo.gl/8S1Oo  
      	"""  
      	# Chi-square statistic (Bevington, eq. 6.9)  
	if sd==None:  
		chisq=np.sum((ydata-ymod)**2)  
	else:  
		chisq=np.sum( ((ydata-ymod)/sd)**2 )  

	return chisq 

def ssvMinFunc(p, gp, sp, ydata, ntrials=1000):
    
    ssdlist=np.arange(.20, .45, .05)
    simdf_list=[]
    sp['mu_ss'] =  -p['mu_ss'].value
    
    for ssd in ssdlist:
        
        sp['ssd']=ssd
        out=ss.set_model(gParams=gp, sParams=sp, mfx=simfx.sim_radd, ntrials=ntrials, timebound=.650, 
                         depHyper=True, visual=False, task='ssRe', return_all_beh=True, condition_str='bsl') 
        simdf_list.append(out)
    
    simdf=pd.concat(simdf_list)
    ymodel=simdf.groupby('ssd').mean()['acc'].values
    
    residuals=ymodel-ydata
    #residuals = scp.sqrt(residuals ** 2 / ysigma ** 2)
    
    return residuals
    
def ssvOpt(params_df, data, ):
	
	for sx, sxdf in params_df.groupby('subj_idx'):
		
		print "%s\n"%str(sx)
		sxdata=data[data['subj_idx']==sx]
		emp_curve=sxdata.groupby('ssd').mean()['acc'].values[:-1]

		params=sxdf['mean']
		params.index=sxdf['param']

		a=params['a']*.1
		z=a*params['z']
		v=params['v(bsl)']*.1

		sp={'mu_ss':-1.0, 'ssTer':0.0, 'ssTer_var':0.0, 'pGo':0.0}
		gp={'a':a, 'v':v, 'z':z, 'Ter':params['t'], 'eta':params['sv'], 'st':0.0, 'sz':0.0}

		p=Parameters()
		p.add('mu_ss', value=.6, min=0.4, max=1.5)
		out = Minimizer(ssvMinFunc, p, fcn_args=(gp,sp,emp_curve), method='Nelder-Mead') 
		out.fmin(maxfun=20)

	return out
