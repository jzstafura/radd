#!/usr/bin/env python
from __future__ import division
from __future__ import division
from radd import ss, psy, simfx
from radd.simfx import sim_radd, sim_ss, sustained_integrator, integrator
import pandas as pd
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


def sim_re(gp, sp, ssdlist=[.4, .35, .3, .25, .20], sim_func=simfx.sim_radd, ntrials=1500, task='ssReBSL', visual=False):

	re_rt=[]; re_p=[]; re_sacc=[]
	#gp={'a':a, 'z':z, 'v':re_v, 'Ter':Ter, 'eta':eta, 'st':st, 'sz':sz}
	
	for ssd in ssdlist:
		
		#sp={'mu_ss':mu_ss, 'pGo':.5, 'ssd':ssd, 'ssTer':ssTer, 'ssTer_var':ssTer_var}
		sp['ssd']=ssd
		
		out=ss.set_model(gParams=gp, sParams=sp, mfx=sim_func, ntrials=ntrials, timebound=tb, visual=visual, task=task)
		
		re_rt.append(out[0]); re_p.append(out[1]); re_sacc.append(out[2])

	df=pd.DataFrame({'goRT': re_rt, 'stopAcc': re_sacc})

	return df


def sim_pro(gp, sp, vlist, pGo_list=[0, .2, .4, .6, .8, 1], sim_func=simfx.sim_radd, ntrials=1500, tb=.550, task='ssProBSL', visual=False, bias_exp=True):
	
	pro_rt=[]; pro_p=[]; pro_sacc=[]
	
	for i,v in enumerate(vlist):
		
		gp['v']=v
		sp['pGo']=pGo_list[i]
		
		out=ss.set_model(gParams=gp, sParams=sp, mfx=sim_func, ntrials=ntrials, timebound=tb, visual=visual, task=task) 
		
		pro_rt.append(out[0]); pro_p.append(out[1]); pro_sacc.append(out[2])	

	df=pd.DataFrame({'goRT': pro_rt, 'pStop': pro_p})

	return df


def plot_evs(bsl, pnl, bsl_GoRT, pnl_GoRT, task='ssRe'):
	
	psy=reload(psy)

	bsl_arr=np.array(bsl, dtype='float')
	pnl_arr=np.array(pnl, dtype='float')
	
	psy.fit_scurves(ysim=[bsl_arr, pnl_arr], task=task)
	
	psy.plot_goRTs(sim_rt=np.array([bsl_GoRT, pnl_GoRT]), task=task)
