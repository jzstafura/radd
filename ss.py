#!/usr/local/bin/env python
from __future__ import division
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from simfx import sim_radd, sim_ss, sustained_integrator, integrator

sns.set(font="Helvetica")

def set_model(gParams=None, sParams=None, mfx=sim_exp, ntrials=100, timebound=0.653, task='ssRe', visual=False, t_exp=False, 
	exp_scale=[10, 10], predictBOLD=False, save=False):

	"""
	set_model: instantiates ddm parameters and call simulation method (mfx)

	args:
		:gParams (dict):		list of dictionaries specifying parameters for go/nogo drift-diffusion signal
		:sParams (dict):		list of dictionaries specifying parameters for stop signal
	
	returns:
		df (pd.DataFrame):		df containing trial-wise results of simulations
								columns=['trial', 'rt', 'choice']
	"""
	
	if gParams is None or sParams is None:
		gp, sp = get_default_parameters()
	else:
		gp=gParams
		sp=sParams
	
	stb=.0001; pStop=1-sp['pGo']
	
	gp, sp = get_intervar_ranges(parameters={'gp':gp, 'sp':sp})
	
	trial_type_list=[]; rt_list=[]; choice_list=[]; acc_list=[]; go_paths_list=[]; go_tsteps_list=[]; ss_paths_list=[]; 
	ss_tsteps_list=[]; len_go_tsteps_list=[]; len_ss_tsteps_list=[]; trial_params_list=[]; thalamus=[]
	
	for i in range(ntrials):
		
		ss_bool=False
		trial_type='go'
		
		if np.random.random_sample()<=pStop:
			ss_bool=True
			trial_type='stop'

		gp['TR'] = gp['Ter_lo'] + np.random.uniform() * (gp['Ter_hi'] - gp['Ter_lo'])
		gp['ZZ'] = gp['z_lo'] + np.random.uniform() * (gp['z_hi'] - gp['z_lo'])
		gp['mu'] = gp['eta'] * np.random.randn() + gp['v']
		tb = stb * np.random.randn() + timebound
		sp['ss_On'] = sp['ssd'] + (sp['ssTer_lo'] + np.random.uniform() * (sp['ssTer_hi'] - sp['ssTer_lo']))

		rt, choice, path, tsteps, ithalamus = mfx(gp['mu'],gp['s2'],gp['TR'],gp['a'],gp['ZZ'], mu_ss=sp['mu_ss'], 
			ssd=sp['ss_On'], timebound=tb, exp_scale=exp_scale, ss_trial=ss_bool, integrate=predictBOLD)	
		
		thalamus.append(ithalamus)

		go_paths_list.append(path[0]); go_tsteps_list.append(tsteps[0]); len_go_tsteps=len(tsteps[0])
		ss_paths_list.append(path[1]); ss_tsteps_list.append(tsteps[1]); len_ss_tsteps=len(tsteps[1])

		if choice==trial_type:
			acc=1
		else:
			acc=0

		rt_list.append(rt); choice_list.append(choice); trial_params_list.append(gp); acc_list.append(acc); 
		len_go_tsteps_list.append(len_go_tsteps); len_ss_tsteps_list.append(len_ss_tsteps); trial_type_list.append(trial_type)

	df=pd.DataFrame({"trial":np.arange(ntrials), "rt":rt_list, "choice":choice_list, "acc":acc_list, 
		"go_tsteps": go_tsteps_list, "go_paths":go_paths_list, "ss_tsteps":ss_tsteps_list, 
		"ss_paths":ss_paths_list, "tparams":trial_params_list, "len_go_tsteps":len_go_tsteps_list, 
		"len_ss_tsteps":len_ss_tsteps_list, "trial_type":trial_type_list})
	
	df_abr=df.drop(['go_tsteps', 'go_paths', 'ss_tsteps', 'ss_paths', 'tparams'], axis=1)
	
	if predictBOLD:
		
		df_bold=df_abr.copy()

		df_bold['thalamus']=thalamus
		
		#df_out=pBOLD(df_bold, task='Re')
		
		return df_bold

	if save:
		savefx(df_abr)

	GoRT, pS, sAcc, GoRT_Err = anl(df_abr)
	sim_data=[GoRT, pS, sAcc, GoRT_Err]

	if visual:
		f=plot_decisions(df=df, pGo=sp['pGo'], timebound=timebound, exp_scale=exp_scale, task=task[:4])
		
		if 'Re' in task:
			savestr="%s_SSD%sms" % (task, str(int(sp['ssd']*1000)))
		else:
			savestr="%s_PGo%s" % (task, str(int(sp['pGo']*100)))

		f.savefig(savestr+".png", format='png', dpi=600)

	return sim_data

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
	
	gp={'a':0.125, 'z':0.063, 'v':0.223, 'Ter':0.435, 'eta':0.133, 'st':0.183, 'sz':0.037, 's2':.01}
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


def anl(df):
	
	GoRT=df.ix[(df['trial_type']=='go')&(df['acc']==1), 'rt'].mean()
	GoRT_Err=df.ix[(df['trial_type']=='stop')&(df['acc']==0), 'rt'].mean() 
	pS=len(df.ix[(df['choice']=='stop')])/len(df)
	sAcc=df.ix[(df['trial_type']=='stop'), 'acc'].mean()
	
	return GoRT, pS, sAcc, GoRT_Err


def plot_decisions(df, pGo=0.5, timebound=0.653, task='ssRe', t_exp=False, exp_scale=[10,10], animate=False):

	plt.ion()
	sns.set(style='white', font="Helvetica")
	
	a=list(pd.Series(df['tparams']))[0]['a']
	z=list(pd.Series(df['tparams']))[0]['z']
	Ter=list(pd.Series(df['tparams']))[0]['Ter']

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
		go_rt=df.ix[(df['trial_type']=='go')&(df['acc']==1), 'rt'].mean()
		go_rt_std=df.ix[(df['trial_type']=='go')&(df['acc']==1), 'rt'].std()
		ssrt=df.ix[(df['trial_type']=='stop')&(df['acc']==1), 'rt'].mean()
		ssrt_std=df.ix[(df['trial_type']=='stop')&(df['acc']==1), 'rt'].std()
		GoLabel='Go Acc'; ssLabel='Stop Acc'
	
	elif 'Pr' in task:
		pG=len(df.ix[(df['choice']=='go')])/len(df)
		pS=len(df.ix[(df['choice']=='stop')])/len(df)
		ssrt=df.ix[(df['choice']=='stop'), 'rt'].mean()
		ssrt_std=df.ix[(df['choice']=='stop'), 'rt'].std()
		go_rt=df.ix[(df['choice']=='go'), 'rt'].mean()
		go_rt_std=df.ix[(df['choice']=='go'), 'rt'].std()
		GoLabel='P(Go)'; ssLabel='P(Stop)'
	
	sns.set_style("white")
	f = plt.figure(figsize=(10,7))
	ax = f.add_subplot(111)

	#xmax=np.array([t for tlist in go_tsteps for t in tlist]).max()
	xmax=timebound+.05
	xmin=Ter-.05
	xlim=ax.set_xlim([xmin, xmax])
	ylim=ax.set_ylim([0, a])
	
	if 'Pr' in task:
		tb_lo=timebound-(1.96*ssrt_std)
		tb_hi=timebound+(1.96*ssrt_std)
		tbrange=np.array([tb_lo, tb_hi])
		tb_low=plt.vlines(x=tb_lo, ymin=0, ymax=a, lw=0.5, color='FireBrick', linestyle='--')
		tb_high=plt.vlines(x=tb_hi, ymin=0, ymax=a, lw=0.5, color='FireBrick', linestyle='--')
		ax.fill_between(tbrange, 0, a, facecolor='Red', alpha=.05)
	
	plt.hlines(y=a, xmin=xlim[0], xmax=xlim[1], lw=4, color='k')
	plt.hlines(y=0, xmin=xlim[0], xmax=xlim[1], lw=4, color='k')
	plt.hlines(y=z, xmin=xlim[0], xmax=xlim[1], alpha=0.5, color='k', lw=4, linestyle='--')
	plt.hlines(y=z, xmin=xlim[0], xmax=Ter, lw=6, color='k', alpha=.5)
	plt.vlines(x=xlim[0], ymin=0, ymax=a, lw=4, color='k')
	plt.vlines(x=timebound, ymin=0, ymax=a, lw=.8, color='Red')
	
	sns.despine(fig=f, ax=ax,top=True, bottom=True, left=False, right=False)

	if 'go' in df.choice.unique():
		clist_go=sns.blend_palette(["#28A428", "#98FFD6"], len(go_paths))
		cycle_go=cycle(clist_go)
	if 'stop' in df.choice.unique():
		clist_ss=sns.blend_palette(["#E60000", "#FF8080"], len(ss_paths))
		cycle_ss=cycle(clist_ss)

	if animate:
		colors=[clist_go, cycle_go]
		animate_paths(ax, go_tsteps, go_paths, ss_tsteps, ss_paths, choices, colors)
	else:
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
			ax.plot(t, p, color=c, alpha=.05, lw=1)

	divider = make_axes_locatable(ax)
	lo = divider.append_axes("bottom", size=1, pad=0, xlim=[xmin, xmax]) #xlim=[0, xmax],
	hi = divider.append_axes("top", size=1, pad=0, xlim=[xmin, xmax])  #xlim=[0, xmax],

	for i, axx in enumerate([hi,lo]):
		if i == 0:
			#df_corr=df.ix[(df['trial_type']=='go')&(df['acc']==1)]
			df_corr=df.ix[(df['choice']=='go')]
			c_corr='LimeGreen'
			c_kde="#2ecc71"
		else:
			#df_corr=df.ix[(df['trial_type']=='stop')&(df['acc']==1)]
			df_corr=df.ix[(df['choice']=='stop')]
			c_corr='Crimson'
			c_kde='Crimson'

		if len(df_corr)<=1: continue
		sns.distplot(np.array(df_corr['rt']), kde=True, ax=axx, kde_kws={"color": c_kde, "shade":True, "lw": 3.5, "alpha":.45},
			hist_kws={"histtype": "stepfilled", "color": c_corr, "alpha":.4});
		#sns.kdeplot(np.array(df_corr['rt']), shade=True, ax=axx, color=c_corr, lw=3.5)

	hi.set_xticklabels([]); hi.set_yticklabels([]); lo.set_yticklabels([]); lo.set_xticklabels([]), lo.invert_yaxis();
	ax.set_xticklabels([]); ax.set_yticklabels([]); f.subplots_adjust(hspace=0)

	hi.text(.07, .8, r'$\mu_{GoRT}=%s$' % (str(go_rt)[:5]), fontsize=14, va='center', ha='left', transform=hi.transAxes, color='Green')
	hi.text(.07, .5, r'$\sigma_{GoRT}=%s$' % (str(go_rt_std)[:5]), fontsize=14, va='center', ha='left', transform=hi.transAxes, color='Green')
	hi.text(.07, .2, r'$%s=%s$' % (GoLabel, str(pG)[:5]), fontsize=14, va='center', ha='left', transform=hi.transAxes, color='Green')
	lo.text(.07, .8, r'$\mu_{ssRT}=%s$' % (str(ssrt)[:5]), fontsize=14, va='center', ha='left', transform=lo.transAxes, color='Red')
	lo.text(.07, .5, r'$\sigma_{ssRT}=%s$' % (str(ssrt_std)[:5]), fontsize=14, va='center', ha='left', transform=lo.transAxes, color='Red')
	lo.text(.07, .2, r'$%s=%s$' % (ssLabel, str(pS)[:5]), fontsize=14, va='center', ha='left', transform=lo.transAxes, color='Red')

	f.suptitle(r'$P(Stop Trial)=%s\,\,&\,\,P(Go Trial)=%s$' % (str(1-pGo), str(pGo)), fontsize=16)

	for a in f.axes:
		a.set_aspect("auto")

	if t_exp:
		plot_timebias_signal(xx, exp_scale, timebound)

	return f

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
