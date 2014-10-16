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



def z_link_func(x, data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return 1 / (1 + np.exp(-(x * stim)))

def v_link_func(x, data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return x * stim

def aic(model):
	k = len(model.get_stochasticts())
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
			m_sx.sample(60000, burn=20000, thin=4)
		
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
	allsx_df.to_csv(save_str+"_sxfits.csv", index=False)
	
	ic_df=pd.DataFrame({'aic':aic_list, 'bic':bic_list, 'dic':dic_list})
	ic_df.to_csv(save_str+"_ic.csv")
	
	return allsx_df


def stats_df(model, save=False):
	"""
	RETURNS:
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
	
	summary_df=pd.pivot_table(param_df, values=['mean', 'std', '2.5q', '25q', 
		'50q', '75q', '97.5q'], index='param', aggfunc=np.average)

	return summary_df


def get_params(task='ssRe'):

	pth=utils.find_path()

	if 'Re' in task:
		params=pd.read_csv(pth+"CoAx/SS/HDDM/Reactive/vbias_full/vBP_SxStats.csv")
		p=ft.stats_summary(params)['mean'].to_dict()
		ssdlist=np.arange(.20, .45, .05)
		vlist=[p['v(bsl)']*.1, p['v(pnl)']*.1]
		conditions=['bsl', 'pnl']
		pGo_list=np.ones(len(vlist))*.5
		tb=.650
	else:
		params=pd.read_csv(pth+"CoAx/SS/HDDM/Proactive/vfull_sx/vfull_HiLo_550_SxStats.csv")
		p=ft.stats_summary(params)['mean'].to_dict()
		ssdlist=np.array([.450])
		vlist=[p['v(Lo)']*.1, p['v(Hi)']*.1]
		conditions=['Lo', 'Hi']
		pGo_list=np.array([.2, .8])
		tb=.560

	#conditions=[c.split('(')[1][:-1] for c in p.keys() if '(' in c]

	a=p['a']*.1; 
	z=p['z']*a; 
	gp={'a':a, 'z':z, 'Ter':p['t'], 'eta':p['sv'], 'st':0.000001, 'sz':0.000001}
	sp={'ssTer':0.0, 'ssTer_var':0.0}

	return gp, sp, vlist, pGo_list, ssdlist, conditions, tb


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
