"""Score deployed stage models on identical, observed real Prefill batches."""
import csv,json,sys,collections
from pathlib import Path
sys.path[:0]=['/host/qwen38-tpu-pp/work/cadence-replay-1113/stage-deps','/host/qwen38-tpu-pp/work/pp-lifecycle-20261112/real-barrier-1113/tools/sglang-simulator/src']
import numpy as np
from sglang_simulator.simulation.vllm.tpu_pp.stage_model import StageModel,StageSpan,PcpLayout,stage_features,tail_features,FEATURE_NAMES,TAIL_FEATURES
root=Path('/host/qwen38-tpu-pp/work/pressure-1116');out=root/'ramp-predictor-1117';cases=json.loads((out/'cases.json').read_text())
paths=[root/'candidate-p0-native-1117/DIAGNOSTIC_ONLY_stage-0_native_return.pkl',Path('/host/qwen38-tpu-pp/work/cadence-replay-1113/pp4_install_artifact/models/DIAGNOSTIC_ONLY_stage-1_native_return.pkl'),Path('/host/qwen38-tpu-pp/work/cadence-replay-1113/pp4_install_artifact/models/DIAGNOSTIC_ONLY_stage-2_native_return.pkl'),root/'candidate-p3-native-r2-1117/DIAGNOSTIC_ONLY_stage-3_native_return.pkl']
models=[StageModel.from_file(str(p),i,3)for i,p in enumerate(paths)];layout=PcpLayout(8,128,(256,512,1024));offset={2048:(5,3,2,-1),4096:(4,5,3,4),8192:(1,3,6,0)}
training={stage:[{k:(v if k in ('case','split') else float(v))for k,v in r.items()} for r in csv.DictReader((root/folder/'predictions.csv').open())if r['split']=='train']for stage,folder in [(0,'candidate-p0-native-1117'),(3,'candidate-p3-native-r2-1117')]}
def metric(rows):
 a=np.array([x['observed_ms']for x in rows]);p=np.array([x['predicted_ms']for x in rows]);d=p-a
 return dict(n=len(rows),observed_mean_ms=float(a.mean()),predicted_mean_ms=float(p.mean()),bias_ms=float(d.mean()),mean_relative_pct=float((p.mean()/a.mean()-1)*100),mae_ms=float(np.abs(d).mean()),wmape_pct=float(np.abs(d).sum()/a.sum()*100),p95_absolute_ms=float(np.quantile(np.abs(d),.95)),observed_p95_ms=float(np.quantile(a,.95)),observed_max_ms=float(a.max()))
def write_csv(path,rows):
 with path.open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[];allrows=[];features=[]
for case in cases:
 n=case['concurrency'];steps=case['steps'];expected={x['step']for x in steps};case_rows=[]
 for x in steps:
  assert all(m['membership_complete']for m in x['members'])
  sp=[StageSpan(m['query_length'],m['past_kv_length'],m['input_length'],False)for m in x['members']]
  f=dict(stage_features(sp,layout),**tail_features(sp));features.append(dict(concurrency=n,step=x['step'],**f));x['features']=f
 for stage,model in enumerate(models):
  source=out/f'c{n}/node{stage}.jsonl';events={}
  for line in source.open():
   e=json.loads(line);sid=e['context'][2]
   assert e['rank']['pp']==stage and e['rank']['pcp']==0 and sid not in events
   assert e['context'][:2]==steps[0]['context'][:2]
   events[sid]=e
  assert set(events)==expected,(n,stage,len(events),len(expected),sorted(expected-set(events))[:5])
  x=np.array([[r['features'][k]for k in model.names]for r in steps]);pred=sum(np.maximum(m.predict(x),0)for m in model.models)*1000
  pred=np.maximum(pred+np.array([offset[r['features']['global_padded_tokens']][stage]for r in steps]),0)
  for i in (0,len(steps)//2,len(steps)-1):
   f=steps[i]['features'];assert abs(pred[i]-max(0,model.predict(f)*1000+offset[f['global_padded_tokens']][stage]))<1e-8
  for row,prediction in zip(steps,pred):
   e=events[row['step']];d=e['data'];actual=(d['observed_mono_ns']-d['start_mono_ns'])/1e6
   assert actual>0 and np.isfinite(actual)
   f=row['features'];case_rows.append(dict(concurrency=n,stage=stage,step=row['step'],batch_size=f['batch_size'],padded_tokens=f['global_padded_tokens'],query_tokens=f['sum_extend'],past_tokens=f['sum_past'],complete_requests=f['prompt_complete_request_count'],observed_ms=actual,predicted_ms=float(prediction),error_ms=float(prediction-actual)))
  rr=[r for r in case_rows if r['stage']==stage];item=dict(concurrency=n,stage=stage,**metric(rr))
  item['by_batch_size']={str(b):metric([r for r in rr if r['batch_size']==b])for b in sorted({r['batch_size']for r in rr})}
  if stage in training:
   train=training[stage];cols=FEATURE_NAMES+(TAIL_FEATURES if stage==3 else [])
   ranges={k:(min(r[k]for r in train),max(r[k]for r in train))for k in cols}
   outside={k:sum(not(lo-1e-9<=r['features'][k]<=hi+1e-9)for r in steps)for k,(lo,hi)in ranges.items()}
   seen={(r['batch_size'],r['global_padded_tokens'])for r in train}
   item['latest_calibration_coverage']=dict(training_n=len(train),outside_feature_range=outside,unseen_batch_padding_count=sum((r['features']['batch_size'],r['features']['global_padded_tokens'])not in seen for r in steps),training_ranges=ranges)
  summary.append(item);print(json.dumps({k:v for k,v in item.items()if k not in ('by_batch_size','latest_calibration_coverage')}),flush=True)
 allrows.extend(case_rows)
write_csv(out/'step-errors.csv',allrows);write_csv(out/'step-features.csv',features)
(out/'step-summary.json').write_text(json.dumps(summary,indent=2));assert len(allrows)==4*sum(len(c['steps'])for c in cases)
print('STEP_EVALUATION_COMPLETE',len(allrows),flush=True)
