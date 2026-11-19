#!/usr/bin/env python3
"""Render the four-case Prefill comparison from the archived summary JSONs."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'qwen38-maxpcp-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    root=args.root
    engine=json.loads((root/'main-engine-summary/engine-summary.json').read_text())
    monitor=json.loads((root/'monitor-summary/summary.json').read_text())
    engine={int(r['concurrency']):r for r in engine}
    monitor={int(r['concurrency']):r for r in monitor}
    conc=[64,128,256,512]
    assert set(engine)==set(conc)==set(monitor)
    assert all(engine[c]['real_success']==engine[c]['paired_successful_requests'] for c in conc)
    successful=sum(engine[c]['real_success'] for c in conc)
    timeout=sum(monitor[c]['real_outcomes']['timeout'] for c in conc)
    assert successful==23340 and timeout==1
    real_color='#2874a6'
    sim_color='#df8c27'
    fail_color='#ba342a'
    error_color='#287b74'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.titlesize':12,'axes.labelsize':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#9aa4ad','axes.linewidth':.7,
                         'pdf.fonttype':42,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(13.8,10.2))
    fig.subplots_adjust(left=.075,right=.975,bottom=.19,top=.83,wspace=.18,hspace=.55)
    fig.text(.075,.965,'Prefill simulation: max-PCP retraining',fontsize=19,weight='bold',ha='left',va='top')
    fig.text(.075,.928,'C64 / C128 / C256 / C512  |  2026-11-19',fontsize=11.5,color='#4d5963',ha='left')
    fig.text(.075,.896,f'{successful:,} successful requests, all paired  |  C512: {timeout} client timeout',
             fontsize=11.5,color='#29343c',ha='left')
    x=np.arange(4)

    def clean(ax):
        ax.set_xticks(x,[f'C{c}' for c in conc])
        ax.tick_params(axis='both',length=0,pad=7)
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#e4e8eb',linewidth=.7)
        ax.set_xlim(-.6,3.6)

    def latency(ax,source,real_key,sim_key,error_key,title,subtitle):
        clean(ax)
        real=np.array([source[c][real_key]/1000 for c in conc])
        sim=np.array([source[c][sim_key]/1000 for c in conc])
        assert np.allclose((sim/real-1)*100,[source[c][error_key] for c in conc])
        ax.bar(x-.18,real,width=.34,color=real_color,label='Real',zorder=3)
        ax.bar(x+.18,sim,width=.34,color=sim_color,label='Simulation',zorder=3)
        for i,(a,b) in enumerate(zip(real,sim)):
            ax.text(i-.18,a+.3,f'{a:.2f}',ha='center',va='bottom',fontsize=9.5,color=real_color)
            ax.text(i+.18,b+.3,f'{b:.2f}',ha='center',va='bottom',fontsize=9.5,color='#a76112')
            error=source[conc[i]][error_key]
            ax.text(i,max(a,b)+1.85,f'{error:+.2f}%',ha='center',va='bottom',fontsize=10,
                    color=fail_color if abs(error)>10 else '#303b43',weight='bold')
        ax.set_ylim(0,30.5)
        ax.set_yticks([0,5,10,15,20,25,30])
        ax.set_ylabel('Mean latency (s)')
        ax.set_title(title,loc='left',pad=37,weight='bold')
        ax.text(0,1.04,subtitle,transform=ax.transAxes,fontsize=9.4,color='#59636c',ha='left')
        ax.legend(loc='upper left',frameon=False,ncol=2,fontsize=9.5)
        ax.get_xticklabels()[0].set_color(fail_color)
        ax.get_xticklabels()[0].set_weight('bold')

    latency(axes[0,0],engine,'real_core','sim_core','engine_ttft_error_pct',
            'A  Prefill engine TTFT','Admission to first-result consumption; successful paired cohort')
    latency(axes[0,1],monitor,'real_deployment_first_ms','sim_deployment_first_ms',
            'deployment_first_ms_error_pct','B  Deployment first-packet latency',
            'Spectrum monitor; 3 matched interior minutes per case')

    def error_panel(ax,key,title,subtitle):
        clean(ax)
        values=np.array([monitor[c][key] for c in conc])
        ax.axhspan(-1,1,color='#ecf4f2',zorder=0)
        ax.axhline(1,color='#648e84',linewidth=1,linestyle=(0,(4,3)),label='Target: +/-1%')
        ax.axhline(-1,color='#648e84',linewidth=1,linestyle=(0,(4,3)))
        ax.axhline(0,color='#9aa8ad',linewidth=.8)
        ax.vlines(x,0,values,color=error_color,linewidth=2,zorder=3)
        ax.scatter(x,values,s=60,color=error_color,zorder=4)
        for i,v in enumerate(values):
            ax.text(i,v+(.105 if v>=0 else -.105),f'{v:+.3f}%',ha='center',
                    va='bottom' if v>=0 else 'top',fontsize=10,weight='bold',color=error_color)
        ax.set_ylim(-1.28,1.28)
        ax.set_yticks([-1,-.5,0,.5,1],['-1.0%','-0.5%','0','+0.5%','+1.0%'])
        ax.set_ylabel('Relative error (%)')
        ax.set_title(title,loc='left',pad=37,weight='bold')
        ax.text(0,1.04,subtitle,transform=ax.transAxes,fontsize=9.4,color='#59636c',ha='left')
        ax.legend(loc='upper right',frameon=False,fontsize=9)

    error_panel(axes[1,0],'INPUT_TPS_error_pct','C  Input throughput error',
                'Spectrum monitor; relative error of input tokens/s')
    error_panel(axes[1,1],'PREFIX_HIT_error_pct','D  Prefix-hit ratio error',
                'Spectrum monitor; relative %, with percentage-point differences in table')
    fig.text(.075,.135,'C64 FAILS the +/-10% TTFT target in both scopes; all cases meet +/-1% input-TPS and prefix targets.',
             color=fail_color,fontsize=10.5,weight='bold',ha='left')
    fig.text(.075,.103,'Error = (simulation / real - 1) x 100%. Engine and deployment panels use different endpoints and sample windows.',
             fontsize=9.5,color='#505d66',ha='left')
    fig.text(.075,.078,'Engine: full successful paired requests. Monitor: 3 interior minutes/case. C512 timeout is excluded from paired engine TTFT.',
             fontsize=9.5,color='#505d66',ha='left')
    fig.text(.075,.053,'Prefix chart shows relative error (%); the accompanying table also reports absolute percentage-point differences (pp).',
             fontsize=9.5,color='#505d66',ha='left')
    for suffix in ('png','pdf'):
        fig.savefig(root/f'results.{suffix}',dpi=180)
    plt.close(fig)
    print(root/'results.png')
    print(root/'results.pdf')


if __name__=='__main__':
    main()
