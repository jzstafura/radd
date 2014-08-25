#!/usr/bin/env python

from __future__ import division
import hddm, os
import numpy as np
import pandas as pd
from myhddm import defmod, parse, vis
from utils import find_path
from patsy import dmatrix


def z_link_func(x, data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return 1 / (1 + np.exp(-(x * stim)))

def v_link_func(x, data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return x * stim

def run_models(mname, project, regress=False):
	
	#bayes fit all subject
	allsx_df=fit_sx(mname, project=project, regress=regress)
	
	#parse model output
	subdf=parse_allsx(allsx_df)
	pdict=subdf_to_pdict(subdf)
	
	#simulate and compare with observed data
	data=defmod.find_data(mname, project=project)
	simdf=vis.predict(pdict, data, ntrials=160, nsims=100, save=True, RTname="SimRT_EvT.jpeg", ACCname="SimACC_EvT.jpeg")
	simdf.to_csv("simdf_sxbayes.csv")
	
	#save pdict; can be reloaded and transformed back into
	#the original pdict format by the following commands 
	#1.  pdict=pd.read_csv("sxbayes_pdict.csv")
	#2.  pdict=pdict.to_dict()
	params=pd.DataFrame(pdict)
	params.to_csv("sxbayes_pdict.csv", index=False)

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
	
def fit_sx(data, model, save_str='x'):
	
	pth=find_path()
	
	grp_dict={}; subj_params=[]; aic_list=[]; bic_list=[]; dic_list=[]; ic_dict={}

	for subj_idx, subj_data in data.groupby('subj_idx'):

		m_sx=(subj_data, depends_on=model.depends_on(), bias=model.bias, include=model.include(), informative=model.informative())
		m_sx.sample(5000, burn=1000, dbname=str(subj_idx)+"_"+save_str+'_traces.db', db='pickle')
		
		sx_df=stats_df(m_sx)
		sx_df=sx_df.drop("sub", axis=1)
		sx_df['sub']=[subj_idx]*len(sx_df)
		
		subj_params.append(sx_df)
		aic_list.append(aic(m_sx)); bic_list.append(bic(m_sx)); dic_list.append(m_sx.dic)

	allsx_df=pd.concat(subj_params)
	allsx_df.to_csv(save_str+"_SxStats.csv", index=False)
	
	ic_dict={'aic':aic_list, 'bic':bic_list, 'dic':dic_list}
	ic_df=pd.DataFrame(ic_dict)
	ic_df.to_csv(save_str+"_IC_Rank.csv")
	
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
	model_stats['sub']=slist
	
	if save:
		model_stats.to_csv('fulldf.csv', index=False)
	
	return model_stats