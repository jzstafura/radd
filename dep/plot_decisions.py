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

def plot_decisions(df, pGo=0.5, ssd=.300, timebound=0.653, task='ssPro', t_exp=False, exp_scale=[10,10], animate=False, normp=False):

	plt.ion()
	sns.set(style='white', font="Helvetica")

	lb=0
	
	ss_tsteps=list(pd.Series(df.ix[(df['choice']=='stop'), 'ss_tsteps']))
	ss_paths=list(pd.Series(df.ix[(df['choice']=='stop'), 'ss_paths']))
	go_tsteps=list(pd.Series(df.ix[(df['choice']=='go'), 'go_tsteps']))
	go_paths=list(pd.Series(df.ix[(df['choice']=='go'), 'go_paths']))
	choices=list(pd.Series(df['choice']))
	
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

	if normp:
		a_orig=a; a=a-z; lb=-z; z=0
	
	#print "a_orig: %s\na: %s\nz: %s\nt: %s" % (str(a_orig), str(a), str(z), str(Ter))
	
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

	f = plt.figure(figsize=(10,7))
	ax = f.add_subplot(111)

	xmax=timebound+.05
	xmin=Ter-.05
	xlim=ax.set_xlim([xmin, xmax])
	ylim=ax.set_ylim([lb, a])
	
	tb_lo=timebound-(1.96*ssrt_std)
	tb_hi=timebound+(1.96*ssrt_std)
	tbrange=np.array([tb_lo, tb_hi])
	tb_low=plt.vlines(x=tb_lo, ymin=lb, ymax=a, lw=0.5, color='FireBrick', linestyle='--')
	tb_high=plt.vlines(x=tb_hi, ymin=lb, ymax=a, lw=0.5, color='FireBrick', linestyle='--')
	ax.fill_between(tbrange, lb, a, facecolor='Red', alpha=.05)
	
	plt.hlines(y=a, xmin=xlim[0], xmax=xlim[1], lw=4, color='k')
	plt.hlines(y=lb, xmin=xlim[0], xmax=xlim[1], lw=4, color='k')
	plt.hlines(y=z, xmin=xlim[0], xmax=xlim[1], alpha=0.5, color='k', lw=4, linestyle='--')
	plt.hlines(y=z, xmin=xlim[0], xmax=Ter, lw=6, color='k', alpha=.5)
	plt.vlines(x=xlim[0], ymin=lb, ymax=a, lw=4, color='k')
	plt.vlines(x=timebound, ymin=lb, ymax=a, lw=.8, color='Red')
	plt.plot(ssd, z, marker='x', ms=15, mew=6, color='Firebrick', alpha=.7)
	
	sns.despine(fig=f, ax=ax,top=True, bottom=True, left=False, right=False)

	clist_go=sns.blend_palette(["#28A428", "#98FFD6"], len(go_paths)+5)
	cycle_go=cycle(clist_go)

	clist_ss=sns.blend_palette(["#E60000", "#FF8080"], len(ss_paths)+5)
	cycle_ss=cycle(clist_ss)

	if animate:
		colors=[clist_go, cycle_go]
		animate_paths(ax, go_tsteps, go_paths, ss_tsteps, ss_paths, choices, colors)
	else:
		for sst, ssp in zip(ss_tsteps, ss_paths):
			if len(ssp)<=1:
				continue
			c=next(cycle_ss)
			ax.plot(sst, ssp, color=c, alpha=.15, lw=1)
		for t,p in zip(go_tsteps, go_paths):
			if len(p)<=1:
				continue
			c=next(cycle_go)
			ax.plot(t, p, color=c, alpha=.05, lw=1)

	divider = make_axes_locatable(ax)
	lo = divider.append_axes("bottom", size=1, pad=0, xlim=[xmin, xmax]) 
	hi = divider.append_axes("top", size=1, pad=0, xlim=[xmin, xmax])  

	for i, axx in enumerate([hi,lo]):
		if i == 0:
			data=df.ix[(df['choice']=='go')]
			c_corr='LimeGreen'
			c_kde="#2ecc71"
		else:
			#data=df.ix[(df['trial_type']=='stop')&(df['acc']==1)]
			data=df.ix[(df['choice']=='stop')]
			c_corr='Crimson'
			c_kde='Crimson'

		if len(data)<=1: continue
		#sns.distplot(np.array(data['rt']), kde=True, ax=axx, kde_kws={"color": c_corr, "shade":True, "lw": 3.5, "alpha":.3},
		#	hist_kws={"histtype": "stepfilled", "color": c_corr, "alpha":.5});
		#sns.kdeplot(np.array(data['rt']), shade=True, ax=axx, color=c_corr, lw=3.5)

	lo.set_xticklabels([]); hi.set_xticklabels([]); ax.set_xticklabels([]);  
	hi.set_yticklabels([]); lo.set_yticklabels([]); ax.set_yticklabels([]); 
	lo.invert_yaxis(); f.subplots_adjust(hspace=0)


	hi.text(.02, .8, r'$\mu_{GoRT}=%s\ (all),\ \ %s\ (correct)$' % (str(go_rt_all)[:5], str(go_rt_cor)[:5]), fontsize=14, transform=hi.transAxes, color='Green')
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
	for a in [hi,lo]:
		a.ymin=lb

	if t_exp:
		plot_timebias_signal(xx, exp_scale, timebound)

	return f
