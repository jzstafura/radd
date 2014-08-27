#!/usr/bin/env python
from __future__ import division
from __future__ import division
from radd import ss, psy, simfx, utils
from radd.simfx import sim_radd, sim_ss, sustained_integrator, integrator, thal
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


def sim_pro(gp, sp, vlist, pGo_list=[0, .2, .4, .6, .8, 1], mfx=simfx.sim_radd, ntrials=1500, return_all=False, return_all_beh=False, tb=.560, task='ssProBSL', visual=False):

	simdf_list=[]
	for i,v in enumerate(vlist):
		
		gp['v']=v
		sp['pGo']=pGo_list[i]
		
		out=ss.set_model(gParams=gp, sParams=sp, mfx=mfx, ntrials=ntrials, timebound=tb, 
			visual=visual, task=task, return_all=return_all, return_all_beh=return_all_beh) 

		simdf_list.append(out)
	
	all_df=pd.concat(simdf_list)

	return all_df


def plot_evs(bsl, pnl, bsl_GoRT, pnl_GoRT, task='ssRe'):
	
	psy=reload(psy)

	bsl_arr=np.array(bsl, dtype='float')
	pnl_arr=np.array(pnl, dtype='float')
	
	psy.fit_scurves(ysim=[bsl_arr, pnl_arr], task=task)
	
	psy.plot_goRTs(sim_rt=np.array([bsl_GoRT, pnl_GoRT]), task=task)



def sim_sx(df, sp=None, pGo_list=[.8,.2], mfx=simfx.sim_radd, ntrials=500, tb=.560, task='ssProBSL', visual=False):

	simdf_list=[]
	if sp is None:
		sp={'mu_ss':-3, 'ssd':.450, 'ssTer':0.0, 'ssTer_var':0.0}

	for sx, sxdf in df.groupby(['subj_idx']):
		
		params=sxdf['mean']
		params.index=sxdf['param']
		a=params['a']*.1
		z=a*params['z']
		vlist=[params['v(Hi)']*.1, params['v(Lo)']*.1]

		gp={'a':a, 'z':z, 'Ter':params['t'], 'eta':params['sv'], 'st':0.000001, 'sz':0.000001}
			
		simdf=sim_pro(gp, sp, vlist=vlist, pGo_list=pGo_list, mfx=mfx, tb=tb, ntrials=ntrials, return_all=True)
		simdf['subj_idx']=[sx]*len(simdf)
		
		simdf_list.append(simdf)

	allsims=pd.concat(simdf_list)
	allsims_abr=allsims.drop(['go_tsteps', 'go_paths', 'ss_tsteps', 'ss_paths', 'tparams'], axis=1)
	

	condition_summaries=[]
	for p, pgo_df in allsims.groupby('pGo'):	
		
		summary=pd.DataFrame([ss.anl(sxdf).T for sxdf in pgo_df.groupby('subj_idx')], index=df['subj_idx'].unique())
		summary['pGo']=[p]*len(summary)
		condition_summaries.append(summary)

		if visual:
			f=ss.plot_decisions(df=pgo_df, pGo=p, ssd=sp['ssd'], timebound=tb, task=task[:4])
			pth=utils.find_path()
			f.savefig(pth+"CoAx/SS/%s_PGo%s.png" % (task, str(int(p*100))), format='png', dpi=600)
	
	pth=utils.find_path()
	allsims_abr.to_csv(pth+"CoAx/SS/HDDM/Proactive/sxfit_raddsims.csv")
	concat_summary=pd.concat(condition_summaries).sort_index()
	concat_summary.to_csv(pth+"CoAx/SS/HDDM/Proactive/sxfit_summary.csv")
	
	return concat_summary



