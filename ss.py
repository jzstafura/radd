#!/usr/local/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import time
from matplotlib.pylab import subplots,close
from matplotlib import cm
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vis2 import visualize_full
from ssex import sim_ssex
from ssex_onsets import sim_true_onsets


def set_model(gParams=None, sParams=None, ntrials=100, timebound=0.653, stb=.0001, task='ssRe', analyze=True, visual=True, animate=False, t_exp=False, true_onsets=False, exp_scale=[10, 10]):

	"""
	gen_model: instantiates ddm parameters and call simulation routine-->sim()

	args:
		:default (Bool):		instantiate model with default parameters (Matzke & Wagenmakers, 2009)

										a=0.125 	# a:    upper boundary (lower boundary is 0)
										z=0.063		# z:    starting point
										nu=0.223 	# nu: 	mean drift rate
										Ter=0.435 	# Ter:	non-decisional time
										eta=0.133 	# eta:	variability in drift rate from trial to trial
										st=0.183  	# st:	variabilit in TR from trial to trial
										sz=0.037  	# sz: 	variability in starting point from trial to trial
										s2=.01  	# s2: 	diffusion coeffient is the amount of within-trial noise

		:plist (list):			list of dictionaries specifying parameters for choice 1 & choice 2
		:visual (Bool):			plot DDM traces with RT distributions on a/b boundaries
		:ntrials (Int):			number of trials to simulate per choice

	returns:
		df (pd.DataFrame):		df containing trial-wise results of simulations
								columns=['trial', 'rt', 'choice']

	__________________PROACTIVE BSL________________ 		_________________PROACTIVE PNL_________________

	SSD = 450ms
	Target Onset = 500ms

                   P(STOP)		              GO RTs
	  p(Go=100) =  0.0046		p(Go=100) =  529.972
	  p(Go=80)  =  0.0344       p(Go=80)  =  534.083
	  p(Go=60)  =  0.2401       p(Go=60)  =  542.921
	  p(Go=40)  =  0.4713       p(Go=40)  =  544.319
	  p(Go=20)  =  0.7438       p(Go=20)  =  545.429
	  p(Go=0)   =  0.9307		p(Go=0)   =  N/A


	_________________REACTIVE BSL_________________ 			_________________REACTIVE PNL_________________

    SSD_200 = 0.994        Mean SSRT = 150.877 ms			SSD_200 = 0.993			Mean SSRT = 138.512 ms
    SSD_250 = 0.982        Mean GoRT = 565.625 ms			SSD_250 = 0.985			Mean GoRT = 573.414769
    SSD_300 = 0.896                 						SSD_300 = 0.925
    SSD_350 = 0.504                 						SSD_350 = 0.594
    SSD_400 = 0.103                 						SSD_400 = 0.181


	"""

	ss_bool=False

	if gParams is None:
		gp={'a':0.24, 'z':0.12, 'v':0.20, 'Ter':0.150, 'eta':0.01, 'st':0.0183, 'sz':0.0137, 's2':.01}
	else:
		gp=gParams

	if sParams is None:
		sp={'mu_ss':-1.0, 'pGo':0.5, 'ssd':.450, 'ssTer':.100, 'ssTer_var':.05}
	else:
		sp=sParams

	gp['Ter_lo'] = gp['Ter'] - gp['st']/2
	gp['Ter_hi'] = gp['Ter'] + gp['st']/2
	gp['z_lo'] = gp['z'] - gp['sz']/2
	gp['z_hi'] = gp['z'] + gp['sz']/2

	sp['ssTer_lo'] = sp['ssTer'] - sp['ssTer_var']/2
	sp['ssTer_hi'] = sp['ssTer'] + sp['ssTer_var']/2
	
	trial_type_list=[]; rt_list=[]; choice_list=[]; acc_list=[];
	go_paths_list=[]; go_tsteps_list=[]; ss_paths_list=[]; ss_tsteps_list=[];
	len_go_tsteps_list=[]; len_ss_tsteps_list=[]; trial_params_list=[]

	pStop=1-sp['pGo']
	
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

		if t_exp and true_onsets:
			rt, choice, path, tsteps = sim_true_onsets(gp['mu'],gp['s2'],gp['TR'],gp['a'],gp['ZZ'], mu_ss=sp['mu_ss'], ssd=sp['ss_On'], timebound=tb, exp_scale=exp_scale, ss_trial=ss_bool)
		
		elif t_exp:
			rt, choice, path, tsteps = sim_ssex(gp['mu'],gp['s2'],gp['TR'],gp['a'],gp['ZZ'], mu_ss=sp['mu_ss'], ssd=sp['ss_On'], timebound=tb, exp_scale=exp_scale, ss_trial=ss_bool)

		else:
			rt, choice, path, tsteps = sim_ss(gp['mu'],gp['s2'],gp['TR'],gp['a'],gp['ZZ'], mu_ss=sp['mu_ss'], ssd=sp['ss_On'], timebound=tb, ss_trial=ss_bool)

		go_paths_list.append(path[0]); go_tsteps_list.append(tsteps[0]); len_go_tsteps=len(tsteps[0])
		ss_paths_list.append(path[1]); ss_tsteps_list.append(tsteps[1]); len_ss_tsteps=len(tsteps[1])

		if choice==trial_type:
			acc=1
		else:
			acc=0

		rt_list.append(rt); choice_list.append(choice); trial_params_list.append(gp); acc_list.append(acc);
		len_go_tsteps_list.append(len_go_tsteps); len_ss_tsteps_list.append(len_ss_tsteps); trial_type_list.append(trial_type)

	df=pd.DataFrame({"trial":np.arange(ntrials), "rt":rt_list, "choice":choice_list, "acc":acc_list, "go_tsteps": go_tsteps_list, "go_paths":go_paths_list,
	"ss_tsteps":ss_tsteps_list, "ss_paths":ss_paths_list, "tparams":trial_params_list, "len_go_tsteps":len_go_tsteps_list, "len_ss_tsteps":len_ss_tsteps_list,
	"trial_type":trial_type_list})

	#df=remove_outliers(df)
	df_abr=df.drop(['go_tsteps', 'go_paths', 'ss_tsteps', 'ss_paths'], axis=1)
	df_abr.to_csv("sims.csv", index=False)

	if analyze:
		GoRT, pS = anl(df_abr, task=task)
		sim_data=[GoRT, pS]

	if visual:
		f=visualize_simple(df=df, pGo=sp['pGo'], timebound=timebound, t_exp=t_exp, exp_scale=exp_scale, task=task, animate=animate)
		plt.savefig("sims%s.png"%i, format='png', dpi=600)

	return sim_data


def sim_ss(mu, s2, TR, a, z, mu_ss=-6, ssd=.450, timebound=0.653, ss_trial=False):

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

	t=TR		        # start the time at TR
 	tau=.0001			# time per step of the diffusion
	choice=None			# init choice as NoneType
	dx=np.sqrt(s2*tau)  # calculate dx (step size)
	e=z;		        # starting point
	e_ss=10000	  		# arbitrary (positive) init value
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
		p_ss=0.5*(1 + mu_ss*dx/s2)

		# if r < p then move up
		if r < p:
			e = e + dx
		# else move down
		else:
			e = e - dx

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


def anl(df, task='ssRe'):
	
	#if task=='ssRe':
		#df_acc=df[df['acc']==1]
		#rt_pivot=pd.pivot_table(df_acc, values='rt', cols=['trial_type'], aggfunc=np.average)
		#acc_pivot=pd.pivot_table(df, values='acc', rows=['trial_type'], aggfunc=np.average)
	
	#elif task=='ssPro':
	#	rt_pivot=pd.pivot_table(df, values='rt', cols=['choice'], aggfunc=np.average)
	#	acc_pivot=pd.pivot_table(df, values='acc', rows=['choice'], aggfunc=np.average)
	
	if task=='ssRe':
		GoRT=df.ix[(df['trial_type']=='go')&(df['acc']==1), 'rt'].mean()
		pS=df.ix[(df['trial_type']=='stop'), 'acc'].mean()
	
	elif task=='ssPro':
		GoRT=df.ix[(df['choice']=='go'), 'rt'].mean()
		pS=len(df.ix[(df['choice']=='stop')])/len(df)
	
	return GoRT, pS


def visualize_simple(df, pGo=0.5, timebound=0.653, task='ssRe', t_exp=False, exp_scale=[10,10], animate=False):

	plt.ion()

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
	
	if task=='ssRe':
		pG=df.ix[(df['trial_type']=='go'), 'acc'].mean()
		pS=df.ix[(df['trial_type']=='stop'), 'acc'].mean()
		go_rt=df.ix[(df['trial_type']=='go')&(df['acc']==1), 'rt'].mean()
		go_rt_std=df.ix[(df['trial_type']=='go')&(df['acc']==1), 'rt'].std()
		ssrt=df.ix[(df['trial_type']=='stop')&(df['acc']==1), 'rt'].mean()
		ssrt_std=df.ix[(df['trial_type']=='stop')&(df['acc']==1), 'rt'].std()
		GoLabel='Go Acc'; ssLabel='Stop Acc'
	
	elif task=='ssPro':
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
	
	if task=='ssPro':
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

	#if t_exp:
	#	xx=np.arange(0, timebound, .0001)
	#	fxx=np.exp(exp_scale[0]*xx)/(np.exp(exp_scale[1]))
	#	expF = plt.figure(figsize=(4, 3))
	#	expAx = expF.add_subplot(111)
	#	expAx.plot(xx, fxx, lw=2, color='k')
	#	expAx.set_ylim(-.0001, .01)
	#	expAx.set_xlim(0, timebound)
	#	expAx.set_xlabel("time (ms)", fontsize=14); expAx.set_ylabel("step-size (dx=.001)", fontsize=14, labelpad=10)
	#	expF.suptitle("urgency signal (+dx)", fontsize=14)

	return f

def remove_outliers(df, sd=2.3):

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