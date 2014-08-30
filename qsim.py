#!/usr/bin/env python
from __future__ import division
from __future__ import division
from radd import ss, psy, simfx, utils
from radd.simfx import sim_radd, sim_ss, sustained_integrator, integrator, thal
import pandas as pd
import numpy as np
import os


#globals
#a=.37; z=.5*a; Ter=.347; eta=.14; st=.0001; sz=.0001; s2=.01; xpo=[12, 12.29]
#mu_ss=-2.17; ssTer=.099; ssRe_TB=.653; ssPro_TB=.6; nt=3000; sTB=.00001; ssTer_var=.0001
#
####ssRe
#vbsl=0.611
#vpnl=0.548
#ssdlist=[.4, .35, .3, .25, .20]
#
####ssPro
#pGo_list=[0, .2, .4, .6, .8, 1]
#pSSD=.450
#vlist_bsl=[0.073, 0.26, 0.4385, 0.601, 0.93, 1.09]
#vlist_pnl=[0.05, 0.23, 0.425, 0.601, 0.93, 1.09]

def simConditions(gp, sp, vlist, pGo_list=[0, .2, .4, .6, .8, 1], ssd=.450, mfx=simfx.sim_radd, ntrials=1500, return_all=False, return_all_beh=False, tb=.560, task='ssProBSL', visual=False, conditions=[]):

	simdf_list=[]

	for i,v in enumerate(vlist):	
		gp['v']=v
		sp['pGo']=pGo_list[i]
		sp['ssd']=ssd

		out=ss.set_model(gParams=gp, sParams=sp, mfx=mfx, ntrials=ntrials, timebound=tb, 
			visual=visual, task=task, return_all=return_all, return_all_beh=return_all_beh, 
			condition_str=conditions[i]) 

		simdf_list.append(out)
	
	
	if len(simdf_list)>1:
		return pd.concat(simdf_list)
	
	else: 
		return simdf_list


def plot_evs(bsl, pnl, bsl_GoRT, pnl_GoRT, task='ssRe'):
	
	psy=reload(psy)

	bsl_arr=np.array(bsl, dtype='float')
	pnl_arr=np.array(pnl, dtype='float')
	
	psy.fit_scurves(ysim=[bsl_arr, pnl_arr], task=task)
	
	psy.plot_goRTs(sim_rt=np.array([bsl_GoRT, pnl_GoRT]), task=task)


def containsAll(strr, sett):
    """ Check whether sequence str contains ALL of the items in set. """
    return 0 not in [c in strr for c in sett]


def sim_sx(df, sp=None, ssdlist=[.450], pGo_list=[], mfx=simfx.sim_radd, ntrials=500, tb=.560, task='ssProBSL'):

	simdf_list=[]
	if sp is None:
		sp={'mu_ss':-1.95, 'ssTer':0.0, 'ssTer_var':0.0}

	for sx, sxdf in df.groupby(['subj_idx']):
		
		params=sxdf['mean']
		params.index=sxdf['param']
		a=params['a']*.1
		z=a*params['z']
		
		vlist=[params[drift]*.1 for drift in params.keys() if containsAll(list(drift), set(["v","("]))] 
		conditions=[c.split('(')[1][:-1] for c in params.keys() if '(' in c]
		
		gp={'a':a, 'z':z, 'Ter':params['t'], 'eta':params['sv'], 'st':0.000001, 'sz':0.000001}
		
		if 'Re' in task or len(pGo_list)==0:
			pGo_list=np.ones(len(vlist))*.5
			ssdlist=np.arange(.20, .45, .05)

		simdf=simConditions(gp, sp, vlist=vlist, pGo_list=pGo_list, mfx=mfx, tb=tb, ntrials=ntrials, 
			conditions=conditions, return_all=True)
		simdf['subj_idx']=[sx]*len(simdf)
		
		simdf_list.append(simdf)

	allsims=pd.concat(simdf_list)
	allsims_abr=allsims.drop(['go_tsteps', 'go_paths', 'ss_tsteps', 'ss_paths', 'tparams'], axis=1)
	
	condition_summaries=[]
	for c, c_df in allsims.groupby('condition'):	
		
		summary=pd.DataFrame([ss.anl(sxdf).T for sxdf in c_df.groupby('subj_idx')], index=df['subj_idx'].unique())
		summary['condition']=[c]*len(summary)
		condition_summaries.append(summary)
	
	sim_summary=pd.concat(condition_summaries).sort_index()
	
	pth=utils.find_path()
	if 'Re' in task:
		sim_summary.to_csv(pth+"CoAx/SS/HDDM/Reactive/sxfit_summary_new.csv")
	else:
		sim_summary.to_csv(pth+"CoAx/SS/HDDM/Proactive/sxfit_summary_new.csv")
	
	return sim_summary



