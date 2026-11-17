"""Saturated frozen real batches through the existing PP workers and LazyPushStepper.

Measures pipeline cost sensitivity, not free-scheduler or client TTFT. Both arms
use precomputed strict lookups, so ML inference CPU overhead cannot confound them.
"""
import csv,json,os,sys,time,dataclasses
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
ROOT=Path('/host/qwen38-tpu-pp/work/pressure-1116/ramp-predictor-1117')
sys.path[:0]=[str(ROOT),'/host/qwen38-tpu-pp/work/cadence-replay-1113/stage-deps','/host/qwen38-tpu-pp/work/pp-lifecycle-20261112/real-barrier-1113/tools/sglang-simulator/src']
from strict_stage_lookup import StrictStageLookup,self_test
from sglang_simulator.simulation.vllm.tpu_pp.stage_model import StageSpan,PcpLayout,stage_features,tail_features,FEATURE_NAMES,TAIL_FEATURES,MODEL_KIND
from sglang_simulator.simulation.vllm.tpu_pp.stage_runtime import VirtualPPExecutor,create_stepper
from sglang_simulator.simulation.vllm.tpu_pp.scheduler_timing import install_scheduler_meta_timing
OFFSETS={2048:(5,3,2,-1),4096:(4,5,3,4),8192:(1,3,6,0)}
@dataclasses.dataclass
class Batch:
    total_num_scheduled_tokens:int=0
    num_scheduled_tokens:dict=dataclasses.field(default_factory=dict)
    scheduled_new_reqs:list=dataclasses.field(default_factory=list)
    scheduled_cached_reqs:object=dataclasses.field(default_factory=lambda:NS(req_ids=[],num_computed_tokens=[]))
    finished_req_ids:set=dataclasses.field(default_factory=set)
    kv_connector_metadata:object=None
class Predictor:
    def __init__(self,paths,config):
        self.stage_model_paths=paths;self.communication_s=config['communication_s'];self.async_handoff=config['async_handoff'];self.barrier_overhead_s=config['barrier_overhead_s'];self.host_prep=config['host_prep'];self.layout=PcpLayout(8,128,(256,512,1024))
    def features_for(self,spans):return stage_features(spans,self.layout),tail_features(spans)
class Inner:
    kv_output_aggregator=True
    ec_output_aggregator=None
    def collective_rpc(self,*args,**kwargs):return None
    def execute_model(self,batch,non_block=False):return NS(ok=True)
    def sample_tokens(self,*args,**kwargs):return None
    def shutdown(self):pass
class FrozenScheduler:
    def __init__(self,steps,metadata):
        self.steps=steps;self.position=0;self.completed={};self.grammar_compile_error_reqs=[]
        install_scheduler_meta_timing(self,metadata,lambda *a,**kw:None)
    def _build_kv_connector_meta(self,connector,output):return None
    def has_requests(self):return self.position<len(self.steps) or len(self.completed)<len(self.steps)
    def get_grammar_bitmask(self,output):return None
    def schedule(self,throttle):
        if self.position==len(self.steps):return Batch()
        row=self.steps[self.position];self.position+=1;members=row['members']
        new=[m for m in members if m['new_request']];cached=[m for m in members if not m['new_request']]
        b=Batch(total_num_scheduled_tokens=sum(m['query_length']for m in members),num_scheduled_tokens={m['request_id_sha256']:m['query_length']for m in members},scheduled_new_reqs=[NS(req_id=m['request_id_sha256'],num_computed_tokens=m['past_kv_length'],prompt_token_ids=range(m['input_length']),sampling_params=NS(logprobs=None))for m in new],scheduled_cached_reqs=NS(req_ids=[m['request_id_sha256']for m in cached],num_computed_tokens=[m['past_kv_length']for m in cached]))
        b._hisim_lifecycle_step_id=row['step'];self._build_kv_connector_meta(None,b);return b
    def update_from_output(self,batch,output):
        if batch.total_num_scheduled_tokens:
            sid=batch._hisim_lifecycle_step_id;assert sid not in self.completed;self.completed[sid]=time.monotonic()
        return {}
def main():
    import joblib
    self_test();n=int(sys.argv[1]);arm=sys.argv[2];limit=int(sys.argv[3])if len(sys.argv)>3 else 0
    assert arm in ('predicted','observed')
    case=next(x for x in json.loads((ROOT/'cases.json').read_text())if x['concurrency']==n);steps=case['steps'][:limit]if limit else case['steps']
    config=json.loads((ROOT/'active-sim-config.json').read_text())['predictor']
    costs={(int(r['stage']),int(r['step'])):r for r in csv.DictReader((ROOT/'step-errors.csv').open())}
    features={int(r['step']):{k:float(v)for k,v in r.items()if k not in ('step','concurrency')}for r in csv.DictReader((ROOT/'step-features.csv').open())if int(r['concurrency'])==n}
    dest=ROOT/('replay-c%d-%s%s'%(n,arm,'-smoke'if limit else ''));dest.mkdir(exist_ok=True);paths=[]
    for stage in range(4):
        names=FEATURE_NAMES+(TAIL_FEATURES if stage==3 else []);x=[[features[r['step']][k]for k in names]for r in steps]
        seconds=[(float(costs[stage,r['step']][arm+'_ms'])-OFFSETS[int(features[r['step']]['global_padded_tokens'])][stage])/1000 for r in steps]
        p=dest/('stage-%d.pkl'%stage);joblib.dump(dict(model_kind=MODEL_KIND,production_usable=False,stage=stage,features=names,time_unit='seconds',models=[StrictStageLookup(x,seconds)]),p);paths.append(str(p))
    os.environ.update(HISIM_TPU_PP_EMULATE_CONNECTOR_QUEUE='1',HISIM_TPU_PP_INITIAL_MEASURED_MS='78.861',HISIM_TPU_PP_PREFILL_HANDOFF_MS='0')
    scheduler=FrozenScheduler(steps,config['scheduler_meta']);executor=VirtualPPExecutor(Inner(),Predictor(paths,config),4)
    executor._req_info={m['request_id_sha256']:(m['input_length'],False)for r in steps for m in r['members']}
    core=NS(model_executor=executor,scheduler=scheduler,_should_throttle_prefills=lambda:False,log_error_detail=lambda b:nullcontext(),capture_iteration_details=lambda b:nullcontext(None),_attach_iteration_details=lambda *a:None,_process_aborts_queue=lambda:None)
    stepper=create_stepper(core,executor);start=time.monotonic();last=0
    try:
        while scheduler.has_requests() or stepper.in_flight or stepper.wave.open or executor.futures_queue:
            if time.monotonic()-start>900:raise TimeoutError('fixed-batch replay')
            stepper.step()
            if len(scheduler.completed)//300>last:
                last=len(scheduler.completed)//300;print(json.dumps(dict(case=n,arm=arm,completed=len(scheduler.completed),elapsed_s=time.monotonic()-start)),flush=True)
            if not stepper.in_flight:time.sleep(.0005)
        end=time.monotonic();assert len(scheduler.completed)==len(steps)==len(executor.step_records)
        records=list(executor.step_records.values());assert all('sample_ready_at'in r for r in records)
        for real,r in zip(steps,records):
            assert r['spans']==[(m['query_length'],m['past_kv_length'])for m in real['members']]
        result=dict(concurrency=n,arm=arm,batches=len(steps),scope='saturated frozen real batches; not free scheduler TTFT',elapsed_s=end-start,first_dispatch_to_last_ready_s=records[-1]['sample_ready_at']-records[0]['dispatched_at'],steps_per_s=len(steps)/(end-start),tokens=sum(r['num_tokens']for r in records),pushes=stepper.wave.pushes,settles=stepper.wave.settles,completion_offsets_s={str(k):v-start for k,v in scheduler.completed.items()})
        (dest/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='completion_offsets_s'}),flush=True)
    finally:
        executor.shutdown();executor._collector.join(timeout=5)
        assert not executor._collector.is_alive()
if __name__=='__main__':main()
