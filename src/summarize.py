"""Aggregate real completed runs, paired inference and figures. Never fills missing runs."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import permutation_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument('--results',default='results'); a=p.parse_args(); root=Path(a.results)/'experiment'; cfg=json.loads(Path('config.json').read_text(encoding='utf-8-sig'))
rows=[]
for m in cfg['models']:
 for s in cfg['seeds']:
  f=root/m/f'seed_{s}'/'metrics.json'
  if not f.exists(): raise RuntimeError(f'Missing evaluation: {f}')
  rows.append(json.loads(f.read_text()))
df=pd.DataFrame(rows)
if df.signature.nunique()!=1: raise RuntimeError('Mixed experimental configurations/manifests')
if df.device.nunique()!=1: raise RuntimeError('Different GPU models: latency comparison requires the same hardware')
if df.training_device.nunique()!=1: raise RuntimeError('Different training GPUs: training-time comparison requires the same hardware. Keep the per-run data and standardize hardware before final aggregation.')
metrics=['accuracy','precision_macro','recall_macro','f1_macro','specificity_macro','auroc_macro_ovr','training_and_validation_seconds','latency_mean_ms','latency_median_ms','latency_p95_ms','parameter_count','peak_gpu_memory_mb']
summary=df.groupby('model')[metrics].agg(['mean','std']); summary.to_csv(root/'summary.csv'); df.to_csv(root/'all_runs.csv',index=False)
wide=df.pivot(index='seed',columns='model',values='f1_macro').sort_index(); differences=wide['yolo26n_cls'].to_numpy()-wide['vgg16'].to_numpy()
# Paired label exchanges by seed; same test split for both algorithms.
test=permutation_test((wide['yolo26n_cls'].to_numpy(),wide['vgg16'].to_numpy()),lambda x,y:np.mean(x-y),permutation_type='samples',vectorized=False,n_resamples=np.inf,alternative='two-sided')
stat=dict(test='Exact paired permutation test of mean macro-F1 difference',direction='YOLO minus VGG',n_pairs=len(wide),mean_difference=float(differences.mean()),p_value=float(test.pvalue),alpha=.05,scope='Training-seed variability conditional on this fixed dataset split; not independent patient-cohort evidence',null_assumption='Within each seed, model outcomes are exchangeable under the null; equivalent sign symmetry of paired differences',conclusion='Evidence of a difference at alpha=0.05' if test.pvalue<.05 else 'Insufficient evidence of a difference at alpha=0.05; this is not equivalence',minimum_possible_two_sided_p=2/(2**len(wide)))
(root/'statistical_test.json').write_text(json.dumps(stat,indent=2),encoding='utf-8'); wide.to_csv(root/'paired_f1.csv')
with pd.ExcelWriter(root/'results.xlsx',engine='openpyxl') as writer:
 df.to_excel(writer,sheet_name='runs',index=False); summary.to_excel(writer,sheet_name='mean_std'); pd.DataFrame([stat]).to_excel(writer,sheet_name='statistical_test',index=False)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for m in cfg['models']:
 histories=[pd.read_csv(root/m/f'seed_{s}'/'history.csv').set_index('epoch') for s in cfg['seeds']]
 loss=pd.concat([h['train_loss'].rename(str(i)) for i,h in enumerate(histories)],axis=1); val=pd.concat([h['val_f1_macro'].rename(str(i)) for i,h in enumerate(histories)],axis=1)
 axs[0].plot(loss.index,loss.mean(axis=1),label=m); axs[1].plot(val.index,val.mean(axis=1),label=m)
 # Per-run curves retained so differing early stopping lengths remain visible.
 for i,h in enumerate(histories):
  fig2,ax=plt.subplots(); ax.plot(h.index,h.train_loss,label='train loss'); ax.plot(h.index,h.val_loss,label='validation loss'); ax.set_xlabel('Epoch'); ax.legend(); fig2.tight_layout(); fig2.savefig(root/m/f'seed_{cfg["seeds"][i]}'/'learning_curve.png',dpi=160); plt.close(fig2)
axs[0].set_ylabel('Mean training loss (available runs)'); axs[1].set_ylabel('Mean validation macro-F1 (available runs)')
for ax in axs: ax.set_xlabel('Epoch'); ax.legend()
fig.tight_layout(); fig.savefig(root/'learning_curves.png',dpi=160); plt.close(fig)
fig,ax=plt.subplots()
for m,g in df.groupby('model'):
 ax.errorbar(g.latency_mean_ms.mean(),g.f1_macro.mean(),xerr=g.latency_mean_ms.std(ddof=1),yerr=g.f1_macro.std(ddof=1),fmt='o',capsize=4,label=m)
ax.set_xlabel('Network-only FP32 latency, batch=1 (ms)'); ax.set_ylabel('Test macro-F1'); ax.legend(); fig.tight_layout(); fig.savefig(root/'performance_latency.png',dpi=160); plt.close(fig)
print(summary.to_string()); print(json.dumps(stat,indent=2)); print('Saved',root/'results.xlsx')
