#!/usr/local/bin/env python
from __future__ import division
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from simfx import sim_radd, sim_ss, sustained_integrator, integrator, sim_ddm, thal, simIndependentPools
import utils

sns.set(font="Helvetica")

def set_model(gParams=None, sParams=None, mfx=sim_radd, ntrials=100, timebound=0.653, s2=.01, task='ssRe', visual=False,
	exp_scale=[12, 12.29], depHyper=True, predictBOLD=False, save=False, return_all=False, return_all_beh=False, condition_str=None):

	"""
	set_model: instantiates ddm parameters and call simulation method (mfx)

	args:
		:gParams (dict):	list of dictionaries specifying parameters for go/nogo drift-diffusion signal
		:sParams (dict):	list of dictionaries specifying parameters for stop signal
	
	returns:
		df (pd.DataFrame):	df containing trial-wise results of simulations
						columns=['trial', 'rt', 'choice']
	"""
	
	if gParams is None or sParams is None:
		gp, sp = get_default_parameters()
	else:
		gp=gParams
		sp=sParams

	stb=.0001; pStop=1-sp['pGo']
	gp, sp = get_intervar_ranges(parameters={'gp':gp, 'sp':sp})
	columns=["rt","choice","acc","go_tsteps", "go_paths","ss_tsteps","thalamus",
		"ss_paths","tparams","len_go_tsteps","len_ss_tsteps","trial_type"]
	df = pd.DataFrame(columns=columns, index=np.arange(0,ntrials))

	for i in range(ntrials):
		
		ss_bool=False
		trial_type='go'
		if np.random.random_sample()<=pStop:
			ss_bool=True
			trial_type='stop'

		gp, sp, tb = update_params(gp, sp, timebound, stb)
		
		rt, choice, paths, tsteps, ithalamus = mfx(gp['mu'], s2, gp['TR'],gp['a'],gp['ZZ'], mu_ss=sp['mu_ss'], ssd=sp['ss_On'], depHyper=depHyper, timebound=tb, exp_scale=exp_scale, ss_trial=ss_bool, integrate=predictBOLD, visual=visual)	
		
		if choice==trial_type: 
			acc=1
		else: 
			acc=0
		
		df.loc[i]=pd.Series({"rt":rt,"choice":choice,"acc":acc,"go_tsteps":tsteps[0], "go_paths":paths[0],"ss_tsteps":tsteps[1],"thalamus":ithalamus,"ss_paths":paths[1],"tparams":gp,"len_go_tsteps":len(tsteps[0]),"len_ss_tsteps":len(tsteps[1]),"trial_type":trial_type})
		
	if condition_str:
		df['condition']=[condition_str]*len(df)
	
	df['ssd']= [sp['ssd']]*len(df)
	df['pGo']=[sp['pGo']]*len(df)

	df_beh=df[['rt', 'choice', 'acc', 'trial_type', 'ssd', 'pGo']]

	if visual:
		
		f=plot_decisions(df=df, pGo=sp['pGo'], ssd=sp['ssd'], timebound=timebound, exp_scale=exp_scale, task=task[:4], normp=False)
		
		if save:
			pth=utils.find_path()
			if 'Re' in task:
				savestr="%s_SSD%sms" % (task, str(int(sp['ssd']*1000)))
			else:
				savestr="%s_PGo%s" % (task, str(int(sp['pGo']*100)))
			f.savefig(pth+"CoAx/SS/"+savestr+".png", format='png', dpi=600)

	if save:
		savefx(df_beh)
	if return_all:
		return df
	if return_all_beh:
		return df_beh
	else:
		sim_data=anl(df_beh)
		return sim_data

def anl(df):
	
	if isinstance(df, tuple):
		indx=df[0]; df=df[1]
	else:
		indx=np.arange(4)

	go_rt_cor=df.ix[(df['trial_type']=='go')&(df['acc']==1), 'rt'].mean()
	go_rt_all=df.ix[(df['trial_type']=='go'), 'rt'].mean()
	go_rt_err=df.ix[(df['trial_type']=='stop')&(df['acc']==0), 'rt'].mean() 
	pstop=len(df.ix[(df['choice']=='stop')])/len(df)
	stop_acc=df.ix[(df['trial_type']=='stop'), 'acc'].mean()
	
	return pd.Series({'go_rt_cor':go_rt_cor, 'go_rt_all':go_rt_all, 'go_rt_err':go_rt_err, 'pstop':pstop, 'stop_acc':stop_acc})

def plot_decisions(df, pGo=0.5, ssd=.300, timebound=0.653, task='ssRe', t_exp=False, exp_scale=[10,10], animate=False, normp=False):

	plt.ion()
	sns.set(style='white', font="Helvetica")
	#try:
	#	a=np.average([xdict['a'] for xdict in list(pd.Series(df['tparams']))])
	#	z=np.average([xdict['z'] for xdict in list(pd.Series(df['tparams']))])
	#	Ter=np.average([xdict['Ter'] for xdict in list(pd.Series(df['tparams']))])
	#	print "list comprehension succeeded: plotting with mean sx params"
	#except Exception:
	#	print "list comprehension failed"
	a=list(pd.Series(df['tparams']))[0]['a']
	z=list(pd.Series(df['tparams']))[0]['z']
	Ter=list(pd.Series(df['tparams']))[0]['Ter']
	lb=0

	if normp:
		a_orig=a; a=a-z; lb=-z; z=0
	
	df_sorted=df.sort(['len_go_tsteps'], ascending=True)
	df_sortGO=df.sort(['len_go_tsteps'], ascending=True)
	df_sortSS=df.sort(['len_ss_tsteps'], ascending=True)
	ss_tsteps=list(pd.Series(df_sortSS['ss_tsteps']))
	ss_paths=list(pd.Series(df_sortSS['ss_paths']))
	go_tsteps=list(pd.Series(df_sortGO['go_tsteps']))
	go_paths=list(pd.Series(df_sortGO['go_paths']))

	choices=list(pd.Series(df_sorted['choice']))
	
	if 'Re' in task:
		pG=df.ix[(df['trial_type']=='go'), 'acc'].mean()
		pS=df.ix[(df['trial_type']=='stop'), 'acc'].mean()
		GoLabel='Go Acc'; ssLabel='Stop Acc'
	
	elif 'Pr' in task:
		pG=len(df.ix[(df['choice']=='go')])/len(df)
		pS=len(df.ix[(df['choice']=='stop')])/len(df)
		GoLabel='P(Go)'; ssLabel='P(Stop)'
	
	go_rt_all=df.ix[(df['choice']=='go'), 'rt'].mean()
	go_rt_cor=df.ix[(df['trial_type']=='go')&(df['acc']==1), 'rt'].mean()
	go_rt_std=df.ix[(df['choice']=='go'), 'rt'].std()
	ssrt=df.ix[(df['choice']=='stop'), 'rt'].mean()
	ssrt_std=df.ix[(df['choice']=='stop'), 'rt'].std()

	f = plt.figure(figsize=(9,7))
	ax = f.add_subplot(111)

	xmax=timebound+.05
	xmin=Ter-.05
	xlim=ax.set_xlim([xmin, xmax])
	ylim=ax.set_ylim([lb, a])
	
	plt.hlines(y=a, xmin=xlim[0], xmax=xlim[1], lw=4, color='k')
	plt.hlines(y=lb, xmin=xlim[0], xmax=xlim[1], lw=4, color='k')
	plt.hlines(y=z, xmin=xlim[0], xmax=xlim[1], alpha=0.5, color='k', lw=4, linestyle='--')
	plt.hlines(y=z, xmin=xlim[0], xmax=Ter, lw=6, color='k', alpha=.5)
	plt.vlines(x=xlim[0], ymin=lb, ymax=a, lw=4, color='k')
	plt.vlines(x=timebound, ymin=lb, ymax=a, lw=.8, color='Red')
	
	sns.despine(fig=f, ax=ax,top=True, bottom=True, left=False, right=False)

	clist_go=sns.blend_palette(["#008140",'#66E0A3'], len(go_paths))
	clist_ss=sns.blend_palette(["#CC0000", "#FFB2B2"], len(ss_paths))	
	cycle_go=cycle(clist_go)
	cycle_ss=cycle(clist_ss)

	for sst, ssp in zip(ss_tsteps, ss_paths):
		if len(sst)<=1:
			continue
		else:
			del sst[0]
			del ssp[0]
		c=next(cycle_ss)
		ax.plot(sst, ssp, color=c, alpha=.15, lw=1)
	for t,p in zip(go_tsteps, go_paths):
		c=next(cycle_go)
		ax.plot(t, p, color=c, alpha=.1, lw=1)

	divider = make_axes_locatable(ax)
	lo = divider.append_axes("bottom", size=1, pad=0, xlim=[xmin, xmax]) #xlim=[0, xmax],
	hi = divider.append_axes("top", size=1, pad=0, xlim=[xmin, xmax])  #xlim=[0, xmax],

	for i, axx in enumerate([hi,lo]):
		if i == 0:
			#df_corr=df.ix[(df['trial_type']=='go')&(df['acc']==1)]
			df_data=df.ix[(df['choice']=='go')]
			c_bins='LimeGreen'
			c_kde="#2ecc71"
		else:
			#df_data=df.ix[(df['trial_type']=='stop')&(df['acc']==1)]
			df_data=df.ix[(df['choice']=='stop')]
			c_bins='#E60000'
			c_kde='#E60000'

		if len(df_data)<=1: continue
		sns.distplot(np.array(df_data['rt']), kde=True, ax=axx, kde_kws={"color": c_kde, "shade":True, "lw": 3.5, "alpha":.45},
			hist_kws={"histtype": "stepfilled", "color": c_bins, "alpha":.4});
		#sns.kdeplot(np.array(df_data['rt']), shade=True, ax=axx, color=c_bins, lw=3.5)

	hi.set_xticklabels([]); hi.set_yticklabels([]); lo.set_yticklabels([]); lo.set_xticklabels([]), lo.invert_yaxis();
	ax.set_xticklabels([]); ax.set_yticklabels([]); f.subplots_adjust(hspace=0)

	hi.text(.02, .8, r'$\mu_{GoRT}=%s\ (all),\ \ %s\ (cor)$' % (str(go_rt_all)[:5], str(go_rt_cor)[:5]), fontsize=14, transform=hi.transAxes, color='Green')
	hi.text(.02, .5, r'$\sigma_{GoRT}=%s$' % (str(go_rt_std)[:5]), fontsize=14, va='center', ha='left', transform=hi.transAxes, color='Green')
	hi.text(.02, .2, r'$%s=%s$' % (GoLabel, str(pG)[:5]), fontsize=14, va='center', ha='left', transform=hi.transAxes, color='Green')
	lo.text(.02, .8, r'$\mu_{ssRT}=%s$' % (str(ssrt)[:5]), fontsize=14, va='center', ha='left', transform=lo.transAxes, color='Red')
	lo.text(.02, .5, r'$\sigma_{ssRT}=%s$' % (str(ssrt_std)[:5]), fontsize=14, va='center', ha='left', transform=lo.transAxes, color='Red')
	lo.text(.02, .2, r'$%s=%s$' % (ssLabel, str(pS)[:5]), fontsize=14, va='center', ha='left', transform=lo.transAxes, color='Red')

	f.suptitle(r'$P(Stop Trial)=%s\,\,&\,\,P(Go Trial)=%s$' % (str(1-pGo), str(pGo)), fontsize=16)

	sns.despine(fig=f, ax=lo, top=True, bottom=True, left=True, right=True)
	sns.despine(fig=f, ax=hi, top=True, bottom=True, left=True, right=True)

	for a in f.axes:
		a.set_aspect("auto")

	return f

def update_params(gp, sp, timebound, stb):
	
	gp['TR'] = gp['Ter_lo'] + np.random.uniform() * (gp['Ter_hi'] - gp['Ter_lo'])
	gp['ZZ'] = gp['z_lo'] + np.random.uniform() * (gp['z_hi'] - gp['z_lo'])
	gp['mu'] = gp['eta'] * np.random.randn() + gp['v']
	tb = stb * np.random.randn() + timebound
	sp['ss_On'] = sp['ssd'] + (sp['ssTer_lo'] + np.random.uniform() * (sp['ssTer_hi'] - sp['ssTer_lo']))

	return gp, sp, tb

def get_intervar_ranges(parameters):
	"""
	:args:
		parameters (dict):	dictionary of gp (Go/NoGo Signal Parameters) 
					and sp (Stop Signal Parameters)
	"""

	gp=parameters['gp']; sp=parameters['sp']

	gp['Ter_lo'] = gp['Ter'] - gp['st']/2
	gp['Ter_hi'] = gp['Ter'] + gp['st']/2
	gp['z_lo'] = gp['z'] - gp['sz']/2
	gp['z_hi'] = gp['z'] + gp['sz']/2

	sp['ssTer_lo'] = sp['ssTer'] - sp['ssTer_var']/2
	sp['ssTer_hi'] = sp['ssTer'] + sp['ssTer_var']/2

	return gp, sp

def get_default_parameters():
	
	"""
	instantiate model with default parameters as found in Matzke & Wagenmakers (2009):
							
		a:   .125  upper boundary (lower boundary is 0)
		z:   .063  starting point
		nu:  .223  mean drift rate
		Ter: .435  non-decisional time
		eta: .133  variability in drift rate from trial to trial
		st:  .183  variabilit in TR from trial to trial
		sz:  .037  variability in starting point from trial to trial
		s2:  .01   diffusion coeffient is the amount of within-trial noise
	
	returns:
		gp:	go/nogo parameters
		sp:	stop signal parameters
	"""
	
	gp={'a':0.125, 'z':0.063, 'v':0.223, 'Ter':0.435, 'eta':0.133, 'st':0.183, 'sz':0.037}
	sp={'mu_ss':-1.0, 'pGo':0.5, 'ssd':.450, 'ssTer':.100, 'ssTer_var':.05}

	return gp, sp

def savefx(abr):

	if 'Re' in task:
		savestr="sims_%s%s%s%s"%(task[2:], '_SSD', str(int(sp['ssd']*1000)), "pGo"+str(int(sp['pGo']*100)))
		task_dir="Reactive/"
	elif 'Pr' in task:
		savestr="sims_%s%s%s%s"%(task[2:], '_PGo', str(int(sp['pGo']*100)), "SSD"+str(int(sp['ssd']*1000)))
		task_dir="Proactive/"
	else:
		savestr="sims"

	if os.path.isdir("/Users/kyle"):
		pth="/Users/kyle/Dropbox/CoAx/ss/simdata/"+task_dir
	elif os.path.isdir("/home/kyle"):
		pth="/home/kyle/Dropbox/CoAx/ss/simdata/"+task_dir
	
	df_abr.to_csv(pth+savestr+'.csv', index=False)
	df.to_csv(pth+savestr+'_full.csv', index=False)


def pBOLD(df):

	dfgo=df.ix[df['trial_type']=='go']
	dfss=df.ix[df['trial_type']=='stop']

	if 'Re' in task:

		gcor=pd.Series(dfgo.ix[dfgo['acc']==1, 'thalamus'], name='CorrectGo')
		gerr=pd.Series(dfgo.ix[dfgo['acc']==0, 'thalamus'], name='IncorrectGo')
		scor=pd.Series(dfss.ix[dfss['acc']==1, 'thalamus'], name='CorrectStop')
		serr=pd.Series(dfss.ix[dfss['acc']==0, 'thalamus'], name='IncorrectStop')
		
		dfout=pd.concat([gcor, gerr, scor, serr], axis=1)
	
	else:
		dfout=df


	return dfout 


def plot_timebias_signal(xx, exp_scale, timebound):
	
	plt.ion()
	xx=np.arange(0, timebound, .0001)
	fxx=np.exp(exp_scale[0]*xx)/(np.exp(exp_scale[1]))
	expF = plt.figure(figsize=(4, 3))
	expAx = expF.add_subplot(111)
	expAx.plot(xx, fxx, lw=2, color='k')
	expAx.set_ylim(-.0001, .01)
	expAx.set_xlim(0, timebound)
	expAx.set_xlabel("time (ms)", fontsize=14); expAx.set_ylabel("step-size (dx=.001)", fontsize=14, labelpad=10)
	expF.suptitle("urgency signal (+dx)", fontsize=14)


def remove_outliers(df, sd=1.95):

	print "len(df) = %s \n\n" % (str(len(df)))

	#print "len(df) = %s \n\n" % (str(len(df)))
	df_ss=df[df['choice']=='stop']
	df_go=df[df['choice']=='go']
	cutoff_go=df_go['rt'].std()*sd + (df_go['rt'].mean())
	df_go_new=df_go[df_go['rt']<cutoff_go]

	#print "cutoff_go = %s \nlen(df_go) = %i\n len(df_go_new) = %i\n" % (str(cutoff_go), len(df_go), len(df_go))

	df_trimmed=pd.concat([df_go_new, df_ss])
	df_trimmed.sort('trial', inplace=True)

	print "cutoff_go = %s \nlen(df_go) = %i\n len(df_go_new) = %i\n" % (str(cutoff_go), len(df_go), len(df_go))

	return df_trimmed


def plot_mean_traces(gomean, ssmean, ssd, Ter=3470):

	sns.set_style('white')
	a=.37; z=.5*a; ssd=ssd+990; 
	f = plt.figure(figsize=(10,7))
	ax = f.add_subplot(111)
	
	if ssd<Ter:
		xlow=ssd-50
	else:
		xlow=Ter-50

	sns.despine(top=True, bottom=True, left=True, right=True)
	
	ax.plot(np.arange(Ter,Ter+len(gomean)), gomean, color='LimeGreen', lw=4)
	ax.plot(np.arange(ssd,ssd+len(ssmean)), ssmean, color='FireBrick', lw=4)
	
	hi=gomean[len(gomean)-1]
	ax.set_xlim([xlow, Ter+len(gomean)+50])
	ax.set_ylim([0, a])#hi+.02
	
	ax.hlines(y=z, xmin=xlow, xmax=Ter+len(gomean)+200, linestyle='--', lw=4, alpha=.5)
	ax.hlines(y=a, xmin=xlow, xmax=Ter+len(gomean)+200, lw=4)
	ax.hlines(y=0, xmin=xlow, xmax=Ter+len(gomean)+200, lw=4)
	ax.vlines(x=xlow, ymin=0, ymax=a, lw=4)
	
	ax.set_yticks(np.arange(0, a, .001))
	ax.set_yticklabels(np.arange(0, a, .01), fontsize=16)
	ax.set_yticklabels([])
	
	ax.set_xticks(np.arange(xlow, Ter+len(gomean)+50, 1))
	#ax.set_xticklabels(np.arange(xlow, Ter+len(gomean), 500), fontsize=16)
	ax.set_xticklabels([])

def concat_output(tasks=None, full=False):

	if tasks is None:
		tasks=['ReBSL_', 'RePNL_']
		conds=['SSD200', 'SSD250', 'SSD300', 'SSD350', 'SSD400']
	else:
		if 'Re' in tasks[0]:
			base_str='ssRe'
			conds=['200ms', '250ms', '300ms', '350ms', '400ms']

		else:
			base_str='ssPro'
			conds=['PGo100', 'PGo80', 'PGo60', 'PGo40', 'PGo20', 'PGo0']
		
		if '_' not in tasks[0]:
			tasks=[t+'_' for t in tasks]
	if full:
		f="_full.csv"
	else:
		f=".csv"

	path='/Users/kyle/Dropbox/git/pynb/'+base_str
	os.chdir(path)
	datasets=[]
	for t in tasks:
		for c in conds:
		    df=pd.read_csv('sims_'+t+c+f)
		    df['Condition']=c
		    df['task']=t.split('_')[0]
		    datasets.append(df)

	alldf=pd.concat(datasets)
	alldf.to_csv(base_str+'_all.csv', index=False)

	return alldf
	

def conditions(cparams, ntrials=100, analyze=True, visual=True, animate=False, t_exp=False, exp_scale=[10, 10]):
	"""
	conditions:		Simulates multiple go/stop decisions based on nested parameter dictionaries, one for each condition.
					Produces multiple figures which are saved to wdir and returns a df with all simulated trials

	args:
		cparams (dict[dicts]):			list of nested dictionaries containing ddm parameters, one dictionary per condition to simulate
										ex: cparams={'Condition1':{'a':0.24, 'z':0.12, 'v':0.20, 'Ter':0.150, 'eta':0.01, 'st':0.0183, 'sz':0.0137, 's2':.01},
														'Condition2':{'a':0.24, 'z':0.12, 'v':0.28, 'Ter':0.450, 'eta':0.01, 'st':0.0183, 'sz':0.0137, 's2':.01}}
		ntrials (int):					number of trials to sim
		analyze (bool):					to return pivot tables for RT and accuracy
		visual	(bool):					to plot simulated diffusion traces
		animate	(bool):					to animate simulated diffusion traces
		t_exp	(bool):					to include an eased exponential time bias
		exp_scale (list):				list w/ 2 elements for scaling the [1] numerator [2] and the denominator of the eased exponential function


	returns:
		simdf (pd.DataFrame):			df containing all simulated data from each condition

	"""

	gpReBSL={'SSD200': { }, 'SSD250': { }, 'SSD300': { }, 'SSD350': { }, 'SSD400': { }}
	gpRePNL={'SSD200': { }, 'SSD250': { }, 'SSD300': { }, 'SSD350': { }, 'SSD400': { }}
	gpPrBSL={'pG100': { }, 'pG80': { }, 'pG60': { }, 'pG40': { }, 'pG20': { }, 'pG0': { }}
	gpPrPNL={'pG100': { }, 'pG80': { }, 'pG60': { }, 'pG40': { }, 'pG20': { }, 'pG0': { }}

	spReBSL={'SSD200': { }, 'SSD250': { }, 'SSD300': { }, 'SSD350': { }, 'SSD400': { }}
	spRePNL={'SSD200': { }, 'SSD250': { }, 'SSD300': { }, 'SSD350': { }, 'SSD400': { }}
	spPrBSL={'pG100': { }, 'pG80': { }, 'pG60': { }, 'pG40': { }, 'pG20': { }, 'pG0': { }}
	spPrPNL={'pG100': { }, 'pG80': { }, 'pG60': { }, 'pG40': { }, 'pG20': { }, 'pG0': { }}

	conds=[[gpReBSL,spReBSL], [gpRePNL, spRePNL], [gpPrBSL, spPrBSL], [gpPrPNL, spPrPNL]]

	#cp=cparams
	for gpsp in conds:

		gp=gpsp[0]
		sp=gpsp[1]

		df=ss.set_model(gParams=gp, sParams=sp, ntrials=200, t_exp=True, exp_scale=[12, 13.4], analyze=False, visual=True)




		params=cp[cond]
