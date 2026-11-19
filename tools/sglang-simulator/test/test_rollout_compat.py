"""Retain baseline frontend and grammar contracts during max-PCP upgrade."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline
from sglang_simulator.simulation.vllm.worker import _constrain_sampled_tokens, _grammar_allowed_token

def test_grammar_chooses_allowed_token():
    row=np.array([1 << 5],dtype=np.int32)
    assert _grammar_allowed_token(row)==5
    assert _grammar_allowed_token(np.array([3],dtype=np.int32))==1
    assert _grammar_allowed_token(np.array([0],dtype=np.int32)) is None

def test_speculative_grammar_rows_follow_request_layout():
    output=SimpleNamespace(req_id_to_index={'a':0,'b':1},sampled_token_ids=[[1],[1]])
    grammar=SimpleNamespace(structured_output_request_ids=['a','b'],grammar_bitmask=np.array([[1<<5],[1<<6],[1<<7]],dtype=np.int32))
    _constrain_sampled_tokens(output,grammar,{'a':[99]})
    assert output.sampled_token_ids==[[5],[7]]

def test_async_frontend_registration_is_early_and_idempotent(monkeypatch):
    seen=[]
    monkeypatch.setattr(pipeline,'_register_tpu_connector_names',lambda:seen.append('register'))
    class Async:
        def __init__(self):seen.append('native_init')
    pipeline.C_VLLMAsyncLLMHook.hook(Async)
    first=Async.__init__
    pipeline.C_VLLMAsyncLLMHook.hook(Async)
    assert Async.__init__ is first
    Async()
    assert seen==['register','native_init']
