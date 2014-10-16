#!/usr/bin/env python
from __future__ import division
from radd import ss, psy, simfx, utils, ft
from radd.simfx import sim_radd, sim_ss, sustained_integrator, integrator, thal
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from progressbar import ProgressBar
from threading import Thread


def simConditions(gp, sp, vlist, pGo_list=[0, .2, .4, .6, .8, 1], ssdlist=[.450], mfx=simfx.sim_radd, depHyper=True, 
	ntrials=1500, return_all=False, return_all_beh=False, tb=.560, task='ssProBSL', visual=False, conditions=[], integrate=False):

	simdf_list=[]
	
	for ssd in ssdlist:
		sp['ssd']=ssd
		
		for i,v in enumerate(vlist):	
			gp['v']=v
			sp['pGo']=pGo_list[i]

			out=ss.set_model(gParams=gp, sParams=sp, mfx=mfx, ntrials=ntrials, timebound=tb, depHyper=depHyper, visual=visual, task=task, return_all=return_all, return_all_beh=return_all_beh, condition_str=conditions[i]) 
			
			out['ssd']=[ssd]*len(out)
			out['condition']=[conditions[i]]*len(out)
			
			simdf_list.append(out)
	
	
	return pd.concat(simdf_list)

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



def fit_sx_ssv(df, ssvlist, task='ssRe', ntrials=20, depHyper=True, nsims=1):

	p=utils.ProgressBar(len(df['subj_idx'].unique()))
	p2=utils.ProgressBar(len(ssvlist))
	p3=utils.ProgressBar(nsims)

	pth = utils.find_path()
	sp={'mu_ss':0.0, 'ssTer':0.0, 'ssTer_var':0.0}

	isx=0
	if 'Re' in task:
		data=pd.read_csv(pth+"CoAx/SS/Reactive/Re_AllData.csv")
	elif 'Pro' in task:
		data=pd.read_csv(pth+"CoAx/SS/Proactive/Pro_AllData.csv")


	fits_df=pd.DataFrame(columns=['subj_idx', 'mse_mu', 'mse_sem'], index=np.arange(len(df['subj_idx'].unique())))
	
	#all_sx_sims=[]
	for sx, sxdf in df.groupby('subj_idx'):
		
		p.animate(isx)
		
		params=sxdf['mean']
		params.index=sxdf['param']
		a=params['a']*.1
		z=a*params['z']

		gp={'a':a, 'z':z, 'Ter':params['t'], 'eta':params['sv'], 'st':0.0, 'sz':0.0}

		if 'Re' in task:
			ssdlist=np.arange(.20, .45, .05)
			#only simulate using bsl params
			vlist=[params['v(bsl)']*.1, params['v(bsl)']*.1]
			pGo_list=np.ones(len(vlist))*.5
			conditions=['bsl', 'bsl']
			tb=.650
		else:
			ssdlist=np.array([.450])
			vlist=[params['v(Lo)']*.1, params['v(Hi)']*.1]
			pGo_list=np.array([.2, .8])
			tb=.560
		
		sxdata=data[data['subj_idx']==sx]
		
		ssvdf_list=[]
		for ssvi, ssv in enumerate(ssvlist):
			
			p2.animate(ssvi)

			sp['mu_ss']=ssv

			nsims_list=[]
			for n in range(nsims):
				
				#p3.animate(n)
				simn=simConditions(gp, sp, vlist, pGo_list=pGo_list, ssdlist=ssdlist, mfx=sim_radd, ntrials=ntrials, depHyper=depHyper, tb=tb, task=task, conditions=conditions, return_all_beh=True, visual=False)

				simn['simn']=[n]*len(simn)
				nsims_list.append(simn)
		
			ssvdf=pd.concat(nsims_list)
			ssvdf['ssv']=[ssv]*len(ssvdf)
			
			ssvdf_list.append(ssvdf)
		
		sx_sims=pd.concat(ssvdf_list)
		sx_sims['subj_idx']=[sx]*len(sx_sims)

		fits=re_mse(sx_sims, sxdata, sx, ssvlist)
		fits_df.loc[isx]=fits
		
		isx+=1

	fits_df.to_csv(pth+'CoAx/SS/ReSSV_Sims/ssv_sxfit_statistics.csv', index=False)

	return fits_df


def fitsss_sx_ssv(df, task='ssRe', ntrials=20, depHyper=True, nsims=1):

	p=utils.ProgressBar(len(df['subj_idx'].unique()))
	p2=utils.ProgressBar(len(ssvlist))
	p3=utils.ProgressBar(nsims)

	pth = utils.find_path()
	sp={'mu_ss':0.0, 'ssTer':0.0, 'ssTer_var':0.0}

	isx=0
	if 'Re' in task:
		data=pd.read_csv(pth+"CoAx/SS/Reactive/Re_AllData.csv")
	elif 'Pro' in task:
		data=pd.read_csv(pth+"CoAx/SS/Proactive/Pro_AllData.csv")


	fits_df=pd.DataFrame(columns=['subj_idx', 'mse_mu', 'mse_sem'], index=np.arange(len(df['subj_idx'].unique())))
	
	#all_sx_sims=[]
	for sx, sxdf in df.groupby('subj_idx'):
		
		p.animate(isx)
		
		params=sxdf['mean']
		params.index=sxdf['param']
		a=params['a']*.1
		z=a*params['z']

		gp={'a':a, 'z':z, 'Ter':params['t'], 'eta':params['sv'], 'st':0.0, 'sz':0.0}

		if 'Re' in task:
			ssdlist=np.arange(.20, .45, .05)
			#only simulate using bsl params
			vlist=[params['v(bsl)']*.1, params['v(bsl)']*.1]
			pGo_list=np.ones(len(vlist))*.5
			conditions=['bsl', 'bsl']
			tb=.650
		else:
			ssdlist=np.array([.450])
			vlist=[params['v(Lo)']*.1, params['v(Hi)']*.1]
			pGo_list=np.array([.2, .8])
			tb=.560
		
		sxdata=data[data['subj_idx']==sx]
		
		ssvdf_list=[]
		for ssvi, ssv in enumerate(ssvlist):
			
			p2.animate(ssvi)

			sp['mu_ss']=ssv

			nsims_list=[]
			for n in range(nsims):
				
				#p3.animate(n)
				simn=simConditions(gp, sp, vlist, pGo_list=pGo_list, ssdlist=ssdlist, mfx=sim_radd, ntrials=ntrials, depHyper=depHyper, tb=tb, task=task, conditions=conditions, return_all_beh=True, visual=False)

				simn['simn']=[n]*len(simn)
				nsims_list.append(simn)
		
			ssvdf=pd.concat(nsims_list)
			ssvdf['ssv']=[ssv]*len(ssvdf)
			
			ssvdf_list.append(ssvdf)
		
		sx_sims=pd.concat(ssvdf_list)
		sx_sims['subj_idx']=[sx]*len(sx_sims)

		fits=re_mse(sx_sims, sxdata, sx, ssvlist)
		fits_df.loc[isx]=fits
		
		isx+=1

	fits_df.to_csv(pth+'CoAx/SS/ReSSV_Sims/ssv_sxfit_statistics.csv', index=False)

	return fits_df


def re_mse(simdf, data, sx, ssvlist):

	sse=[]
	mse_mu=[]
	mse_sem=[]
	stop_curves=[]

	simdf['acc']=simdf['acc'].astype(float)


	emp_curve=data.groupby('ssd').mean()['acc'].values[:-1][::-1]
	#print "sx: %s\n pstop: %s" % (str(sx), str(emp_curve))
	for ssv, ssv_df in simdf.groupby('ssv'):
		ptable = utils.makePivot(ssv_df.query('trial_type=="stop" & condition=="bsl"'), cols='simn', index='ssd', values='acc')
		
		for c in ptable.columns:
			sse.append(np.sum((ptable[c]-emp_curve[::-1])**2))

		mse_mu.append(np.average(sse))
		mse_sem.append(np.std(sse)/np.sqrt(len(sse)))
		stop_curves.append(ptable.mean(axis=1).values[::-1])
	
	sns.set_context('poster')
	sns.set_style('white')
	f, ax = plt.subplots(1, figsize=(7, 5))
	labels=[str(ssv) for ssv in ssvlist]
	
	ax = psy.basic_curves(ysim=stop_curves[::-1], task='ssRe', showPSE=True, ax=ax, labels=labels)
	ax = psy.basic_curves(ysim=[emp_curve], task='ssRe', showPSE=True, ax=ax, labels=['bsl'], line_colors=['Gray'], point_colors=sns.blend_palette(["Purple", "Purple"]), show_emp_line=True)
	pth=utils.find_path()
	plt.savefig(pth+'CoAx/SS/ReSSV_Sims/sx'+str(sx)+"_StopCurves_bySSV.png", format='png', dpi=300)

	fits=pd.Series({'subj_idx':sx, 'mse_mu':mse_mu, 'mse_sem':mse_sem})

	return fits

def sim_ssv_range(ssvlist=-np.arange(.5, 2.5, .5), params=None, task='ssRe', mfx=simfx.thal, ntrials=200, depHyper=True, visual=False):

	sims=[]
	gp, sp, vlist, pGo_list, ssdlist, conditions, tb=get_params(task)

	p=utils.ProgressBar(len(ssvlist))

	for i, ssv in enumerate(ssvlist):
		
		p.animate(i)

		sp['mu_ss']=ssv
		simdf=simConditions(gp, sp, vlist, pGo_list=pGo_list, ssdlist=ssdlist, mfx=mfx, ntrials=ntrials, depHyper=depHyper, tb=tb, task='ssRe', conditions=conditions, return_all_beh=True, visual=visual)
					
		simdf['ssv']=[ssv]*len(simdf)
		sims.append(simdf)

		#print "Finished %s of %s" % (str(i+1), str(len(ssvlist)))

	allSims=pd.concat(sims)    

	return allSims

def plot_evs(bsl, pnl, bsl_GoRT, pnl_GoRT, task='ssRe'):
	
	psy=reload(psy)

	bsl_arr=np.array(bsl, dtype='float')
	pnl_arr=np.array(pnl, dtype='float')
	
	psy.fit_scurves(ysim=[bsl_arr, pnl_arr], task=task)
	
	psy.plot_goRTs(sim_rt=np.array([bsl_GoRT, pnl_GoRT]), task=task)


def containsAll(strr, sett):
    """ Check whether sequence str contains ALL of the items in set. """
    return 0 not in [c in strr for c in sett]


def sim_sx(df, sp=None, ssdlist=[.450], pGo_list=[], mfx=simfx.thal, ntrials=500, tb=.560, task='ssProBSL', summarize=True):

	simdf_list=[]
	if sp is None:
		sp={'mu_ss':-1.95, 'ssTer':0.0, 'ssTer_var':0.0}

	for sx, sxdf in df.groupby(['subj_idx']):
		
		params=sxdf['mean']
		params.index=sxdf['param']
		a=params['a']*.1
		z=a*params['z']
		
		#vlist=[params[drift]*.1 for drift in params.keys() if containsAll(list(drift), set(["v","("]))] 
		conditions=[c.split('(')[1][:-1] for c in params.keys() if '(' in c]
		#vlist=np.sort(np.array(vlist))
		gp={'a':a, 'z':z, 'Ter':params['t'], 'eta':params['sv'], 'st':0.0, 'sz':0.0}

		if 'Re' in task or len(pGo_list)==0:
			ssdlist=np.arange(.20, .45, .05)
			pGo_list=np.ones(len(vlist))*.5
			vlist=[params['v(bsl)']*.1, params['v(pnl)']*.1]

		else:
			ssdlist=np.array([.450])
			vlist=[params['v(Lo)']*.1, params['v(Hi)']*.1]
			pGo_list=np.array([.2, .8])
		
		simdf=simConditions(gp, sp, vlist=vlist, ssdlist=ssdlist, pGo_list=pGo_list, mfx=mfx, tb=tb, ntrials=ntrials, conditions=conditions, return_all_beh=True)
		simdf['subj_idx']=[sx]*len(simdf)
		
		simdf_list.append(simdf)

	allsims=pd.concat(simdf_list)

	if not summarize:
		return allsims

	else:
		condition_summaries=[]
		for c, c_df in allsims.groupby(['ssd', 'condition']):	
			
			summary=pd.DataFrame([ss.anl(sxdf).T for sxdf in c_df.groupby('subj_idx')], index=df['subj_idx'].unique())
			
			summary['ssd']=[c[0]]*len(summary)
			summary['condition']=[c[1]]*len(summary)
			
			condition_summaries.append(summary)
	
		sim_summary=pd.concat(condition_summaries).sort_index()
	
		pth=utils.find_path()
		if 'Re' in task:
			sim_summary.to_csv(pth+"CoAx/SS/HDDM/Reactive/sxfit_summary_search.csv")
		else:
			sim_summary.to_csv(pth+"CoAx/SS/HDDM/Proactive/sxfit_summary_new.csv")
	
		return sim_summary



