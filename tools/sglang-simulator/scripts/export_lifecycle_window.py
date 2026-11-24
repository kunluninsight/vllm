"""Lossless selected-event export from all PP/PCP processes via allowed read commands.
No rank0 sampling, no 30-second thinning, no missing result treated as empty success.
"""
import argparse,collections,gzip,hashlib,json,math,re,subprocess,time,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from sglang_simulator.lifecycle_event_kinds import LIFECYCLE_EVENT_KINDS
ROOT=Path(os.environ.get('PP_RETRAIN_ROOT',str(Path(__file__).resolve().parent)))
BASE=os.environ.get('PP_EXPORT_TRACE_BASE','/home/admin/latency_collection_codex-pp-lifecycle-r2-20261111-aa73f1d7/trace_v3')
WORKSPACE=os.environ.get('PP_EXPORT_WORKSPACE','test')
last=0
KINDS=['request.input_processed','request.identity_alias','request.business_external_alias','request.add','request.scheduler_add','request.kv_lookup','request.external_match','request.remote_kv_promoted','request.step_membership','request.output_materialization','request.logical_finish','rpc.execute_model.dispatch','pipeline.result_ready','pp.dispatch','pp.sample_ready','scheduler.scheduled','scheduler.output_update','native.forward_dispatch_enter','native.forward_dispatch_return','native.local_drain.enter','native.local_drain.ready','native.output_ready_observed','native.handoff_barrier_done','native.handoff_submitted','native.completion_unresolved','chunk.inputs']
KINDS=sorted(set(KINDS) | LIFECYCLE_EVENT_KINDS)
SELECTION=hashlib.sha256(json.dumps(KINDS).encode()).hexdigest()

def call(pod,*cmd):
 global last
 for attempt in range(8):
  time.sleep(max(0,5-(time.monotonic()-last)))
  p=subprocess.run(['dashctl','-w',WORKSPACE,'exec','inst/'+pod,'--',*cmd],capture_output=True,text=True,timeout=180);last=time.monotonic()
  if p.returncode and ('429'in p.stderr or 'NCS login check timed out'in p.stderr):
   time.sleep(min(120,15*2**attempt));continue
  if p.returncode:
   # grep=1 may mean no matches but ASI hides the exit code: never silently accept it.
   raise RuntimeError(p.stderr[:600])
  return p.stdout.replace('\\n','\n') if '\n' not in p.stdout.strip() else p.stdout
 raise RuntimeError('Read retry exhaustion')

def grep_records(pod,files,lo,hi):
 cache=ROOT/'read_cache'/hashlib.sha256(json.dumps([pod,files,lo,hi,KINDS,'v5-lifecycle-query-job'],sort_keys=True).encode()).hexdigest()
 cache.parent.mkdir(exist_ok=True)
 if cache.with_suffix('.jsonl.gz').exists():
  with gzip.open(cache.with_suffix('.jsonl.gz'),'rt')as f:return list(map(json.loads,f))
 seconds='|'.join(str(t)for t in range(lo,hi+1))
 native=[k for k in KINDS if k.startswith('native.')or k=='chunk.inputs']
 control=[k for k in KINDS if k not in native and k!='step.inputs']
 pattern='"mono_ns":[[:space:]]*('+seconds+')[0-9]{9},.*(("event":[[:space:]]*"('+'|'.join(re.escape(x)for x in native)+')".*"context":[[:space:]]*\\[[^]]*"real"\\])|("event":[[:space:]]*"('+'|'.join(re.escape(x)for x in control)+')")|("event":[[:space:]]*"step.inputs".*"total_num_scheduled_tokens":[[:space:]]*[1-9]))'
 try:
  text=call(pod,'grep','-EH','('+pattern+')|^PRETTY_NAME=',*files,'/etc/os-release')
  result=[];sentinel=False
  for line in text.splitlines():
   if line.startswith('/etc/os-release:PRETTY_NAME='):
    sentinel=True;continue
   if line.strip():
    source,sep,body=line.partition(':{')
    if not sep:raise json.JSONDecodeError('Missing source path provenance',line,0)
    e=json.loads('{'+body);e['_source_file']=source
    result.append(e)
  if not sentinel:raise json.JSONDecodeError('End-of-response sentinel missing; possible truncation',text,len(text))
  with gzip.open(cache.with_suffix('.jsonl.gz'),'wt')as f:
   for e in result:f.write(json.dumps(e,separators=(',',':'))+'\n')
  return result
 except json.JSONDecodeError as e:
  if 'text'in locals():
   debug=ROOT/'private'/('read_error_'+hashlib.sha256((pod+str(files)+str(lo)+str(hi)).encode()).hexdigest()[:16])
   debug.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
   debug.with_suffix('.txt').write_text(text)
   debug.with_suffix('.json').write_text(json.dumps({'pod':pod,'files':files,'lo':lo,'hi':hi,'error':str(e)}))
  if hi-lo>2:
   middle=(lo+hi)//2
   return grep_records(pod,files,lo,middle)+grep_records(pod,files,middle+1,hi)
  if len(files)>1:
   middle=len(files)//2
   return grep_records(pod,files[:middle],lo,hi)+grep_records(pod,files[middle:],lo,hi)
  raise RuntimeError(f'Cannot prove complete read for {pod} {lo}-{hi}: {e}')from e

def main():
 global KINDS
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--start',type=float,required=True);ap.add_argument('--end',type=float,required=True);ap.add_argument('--group');a=ap.parse_args()
 if not(ROOT/'injection_complete.json').exists():raise RuntimeError('Missing injection inventory; capture current process metadata first')
 inventory=sorted(json.loads((ROOT/'injection_complete.json').read_text()),key=lambda x:('.rank-0-'not in x['pod'],x['pod']))
 target=ROOT/'raw'/a.case;target.mkdir(parents=True,exist_ok=True)
 report=[]
 for host in inventory:
  pod=host['pod']
  if a.group and a.group not in pod:continue
  dest=target/pod;dest.mkdir(exist_ok=True)
  if(dest/'receipt.json').exists():
   prior=json.loads((dest/'receipt.json').read_text())
   assert prior.get('selection_sha256')==SELECTION,'Old selection receipt: choose a fresh case/output directory'
   report.append(prior);continue
  metadata=json.loads((ROOT/'injection'/pod/'metadata.json').read_text())
  offsets=[(m[0]['wall_anchor_ns']-m[0]['mono_anchor_ns'])/1e9 for m in metadata.values()]
  lo=math.floor(a.start-max(offsets))-3;hi=math.ceil(a.end-min(offsets))+3
  paths=call(pod,'find',BASE+'/'+pod,'-maxdepth','6','-type','f','-name','events.*.jsonl','-newermt',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(a.start-10))).splitlines()
  paths=[p for p in paths if p.startswith(BASE+'/'+pod+'/')]
  assert paths,('No files',pod)
  (dest/'source_paths.json').write_text(json.dumps(paths,indent=2))
  # Segment by file if response would exceed transport limits; all files remain accounted for.
  # Source process is known from metadata; read it directly instead of a binary
  # tree of repeatedly truncated all-worker responses. Completed subreads are cached.
  by_process={}
  for path in paths:
   proc=next((str(Path(m).parent)for m in metadata if path.startswith(str(Path(m).parent)+'/')),None)
   assert proc,('No source metadata',path)
   by_process.setdefault(proc,[]).append(path)
  rows=[]
  def pid_order(item):
   return int(Path(item[0]).name.split('-')[0])
  for proc,source_files in sorted(by_process.items(),key=pid_order):
   block=grep_records(pod,source_files,lo,hi);rows.extend(block)
   one=dest/('process-'+Path(proc).name+'.jsonl.gz')
   with gzip.open(one,'wt')as out:
    for e in block:out.write(json.dumps(e,separators=(',',':'))+'\n')
   info={'pod':pod,'process':proc,'source_files':source_files,'mono_window':[lo,hi],'events':len(block),'event_counts':dict(collections.Counter(e['event']for e in block)),'path':str(one),'sha256':hashlib.sha256(one.read_bytes()).hexdigest(),'process_read_complete':True}
   one.with_suffix('.receipt.json').write_text(json.dumps(info,indent=2));print(json.dumps({'process_complete':proc,'events':len(block),'event_counts':info['event_counts']}),flush=True)
  unique={}
  for e in rows:
   # seq is local to process: preserve process identity from original source path? Raw event has pid?
   # Canonical complete JSON includes rank/context/mono/seq. Identical duplicate transport reads only are removed.
   proc=next((str(Path(m).parent)for m in metadata if e['_source_file'].startswith(str(Path(m).parent)+'/')),None)
   assert proc,('Missing metadata for source file',e['_source_file'])
   e['_source_process_dir']=proc
   key=(proc,e['seq'])
   if key in unique:
    prior={k:v for k,v in unique[key].items()if k!='_source_file'}
    curr={k:v for k,v in e.items()if k!='_source_file'}
    assert prior==curr,('Conflicting process sequence',key)
   unique[key]=e
  rows=sorted(unique.values(),key=lambda e:(e.get('mono_ns',0),e.get('seq',0)))
  path=dest/'events.jsonl.gz'
  with gzip.open(path,'wt')as out:
   for e in rows:out.write(json.dumps(e,separators=(',',':'))+'\n')
  (dest/'metadata.json').write_text(json.dumps(metadata,indent=2))
  r={'selection_sha256':SELECTION,'selected_events':KINDS,'pod':pod,'requested_wall':[a.start,a.end],'mono_seconds_with_margin':[lo,hi],'files':len(paths),'events':len(rows),'event_counts':dict(collections.Counter(e['event']for e in rows)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'full_selected_window':True,'no_rank_or_time_sampling':True,'selection':'all request/controller events including zero-token RPC; nonzero step.inputs; all real-context chunk/native events; native padding-only context-null protocol points remain on source'}
  (dest/'receipt.json').write_text(json.dumps(r,indent=2));report.append(r)
  (target/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(r),flush=True)
 (target/'complete.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
