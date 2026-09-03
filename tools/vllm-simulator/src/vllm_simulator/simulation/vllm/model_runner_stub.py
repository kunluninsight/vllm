"""
ModelRunner stub - CPU replacement for GPUModelRunner.

Implements the model_runner-side contract the native Worker drives:
get_kv_cache_spec / initialize_kv_cache / execute_model / sample_tokens.
Physical KV allocation is MINIMAL (1 page per tensor) and page sizes are
accounting-only, so they stay at real model values (no scale factor).
"""

import dataclasses

import torch

from vllm_simulator.simulation.manager import ConfigManager
from vllm_simulator.utils import get_logger

logger = get_logger()


def _noop(*args, **kwargs):
    """Module-level no-op function (picklable)."""
    return None


class _ModelRunnerStub:
    """Picklable stub for model_runner (must be module-level for multiprocess).

    Everything not explicitly implemented resolves to a no-op via
    __getattr__.
    """

    # Version-probed once at hook-install time by C_VLLMWorkerHook.
    _MRO_FIELDS: frozenset[str] = frozenset()
    _KVConnectorOutput = None

    def __init__(self, kv_spec=None):
        self._kv_spec = kv_spec
        self.model_memory_usage = 0
        # Explicit None so LLMEngine._get_driver_model_for_cleanup() sees no
        # model and skips its bytecode-cleanup finalizer (the __getattr__
        # fallback would otherwise return a no-op callable here).
        self.model = None
        self._last_model_output = None

    def get_kv_cache_spec(self):
        return self._kv_spec

    def initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        """CPU stand-in for GPUModelRunner.initialize_kv_cache.

        Native Worker.initialize_from_config calls this after connector
        creation: build the minimal CPU tensors and register them with the
        KV connector — the two duties of the real runner at this point.
        """
        from vllm.distributed.kv_transfer import (
            has_kv_transfer_group,
            get_kv_transfer_group,
        )

        num_blocks = kv_cache_config.num_blocks
        kv_caches: dict = {}
        # CPU simulation: allocate MINIMAL tensors (1 page each) to avoid OOM.
        # The full num_blocks count is preserved for scheduling/prefix logic,
        # but we don't need actual KV data storage in simulation mode.
        if getattr(kv_cache_config, "kv_cache_tensors", None):
            for kv_tensor in kv_cache_config.kv_cache_tensors:
                # Allocate only 1 page instead of full size
                minimal_size = min(kv_tensor.size, 4096)
                tensor = torch.zeros(
                    minimal_size, dtype=torch.int8, device="cpu"
                )
                # `shared_by` was renamed to `layers` in newer vLLM
                layer_names = getattr(kv_tensor, "layers", None)
                if layer_names is None:
                    layer_names = kv_tensor.shared_by
                for layer_name in layer_names:
                    kv_caches[layer_name] = tensor
            logger.info(
                "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
                "(num_blocks=%d, simulated_bytes=%d, actual_bytes=4k)",
                len(kv_cache_config.kv_cache_tensors), num_blocks,
                sum(t.size for t in kv_cache_config.kv_cache_tensors),
            )
        else:
            for layer_name, spec in self._kv_spec.items():
                # Allocate only 1 page instead of num_blocks * page_size
                minimal_size = min(spec.page_size_bytes, 4096)
                tensor = torch.zeros(
                    minimal_size, dtype=torch.int8, device="cpu"
                )
                kv_caches[layer_name] = tensor
            logger.info(
                "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
                "(num_blocks=%d, page_size=%d, actual_alloc=4k)",
                len(kv_caches), num_blocks,
                spec.page_size_bytes if self._kv_spec else 0,
            )

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)
            logger.info("[vLLM Hijack] KV caches registered with connector")

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        """CPU stand-in for GPUModelRunner.execute_model.

        The native Worker.execute_model delegates here (tp=pp=1).  The KV
        connector lifecycle around the mock forward mirrors what
        KVConnectorModelRunnerMixin does for the real runner.
        """
        from vllm.distributed.kv_transfer import (
            has_kv_transfer_group,
            get_kv_transfer_group,
        )
        from vllm.v1.outputs import ModelRunnerOutput

        # KV connector pre-forward
        kv_connector = None
        try:
            if has_kv_transfer_group():
                kv_connector = get_kv_transfer_group()
                kv_connector_metadata = scheduler_output.kv_connector_metadata
                if kv_connector_metadata is not None:
                    if hasattr(kv_connector, "handle_preemptions"):
                        kv_connector.handle_preemptions(kv_connector_metadata)
                    kv_connector.bind_connector_metadata(kv_connector_metadata)
                    kv_connector.start_load_kv(None)
        except Exception:
            logger.exception(
                "[vLLM Hijack] execute_model: KV pre-forward lifecycle failed"
            )
            kv_connector = None

        # Mock forward.  A chunked-prefill step only produces a sampled
        # token when the request has reached the end of its prompt; the
        # per-request decision is computed by the executor hook
        # (C_VLLMExecutorHook) and annotated onto scheduler_output
        # (_sim_token_emitted), riding the native scheduler -> worker
        # dataflow.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        req_ids = list(num_scheduled_tokens) if num_scheduled_tokens else []
        token_emitted = getattr(scheduler_output, "_sim_token_emitted", None)
        if req_ids and token_emitted is None:
            raise RuntimeError(
                "scheduler_output lacks _sim_token_emitted annotation "
                "(sim executor hook not active?)"
            )

        sampled_token_ids = []
        for req_id in req_ids:
            if req_id not in token_emitted:
                raise RuntimeError(
                    f"scheduled request {req_id!r} is missing from the "
                    "executor hook's _sim_token_emitted annotation"
                )
            sampled_token_ids.append([1] if token_emitted[req_id] else [])

        mro_kwargs = dict(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
        )
        if "kv_lens" in self._MRO_FIELDS:
            mro_kwargs["kv_lens"] = [0] * len(req_ids)
        if "pooler_output" in self._MRO_FIELDS:
            mro_kwargs["pooler_output"] = [None] * len(req_ids)
        output = ModelRunnerOutput(**mro_kwargs)

        # KV connector post-forward
        if kv_connector is not None and self._KVConnectorOutput is not None:
            try:
                kv_connector.wait_for_save()
                kv_output = self._KVConnectorOutput()
                finished_req_ids = getattr(scheduler_output, "finished_req_ids", set())
                kv_output.finished_sending, kv_output.finished_recving = (
                    kv_connector.get_finished(finished_req_ids)
                )
                if kv_output.finished_sending or kv_output.finished_recving:
                    logger.info(
                        "[vLLM Hijack] execute_model: finished_sending=%s "
                        "finished_recving=%s",
                        sorted(kv_output.finished_sending or []),
                        sorted(kv_output.finished_recving or []),
                    )
                if hasattr(kv_connector, "build_connector_worker_meta"):
                    kv_output.kv_connector_worker_meta = (
                        kv_connector.build_connector_worker_meta()
                    )
                kv_connector.clear_connector_metadata()
                output.kv_connector_output = kv_output
            except Exception:
                logger.exception(
                    "[vLLM Hijack] execute_model: KV post-forward lifecycle failed"
                )

        self._last_model_output = output
        return output

    def sample_tokens(self, grammar_output):
        """Fallback for the async-sampling path: execute_model never returns
        None here, so this only satisfies the native Worker delegation."""
        return self._last_model_output

    def __getattr__(self, name):
        # Any attribute not explicitly defined returns a no-op callable
        if name.startswith('_'):
            raise AttributeError(name)
        return _noop


def _native_meta_spec(vllm_config) -> "dict | None":
    """Derive the KV cache spec the way the real GPUModelRunner does:
    build the model on the meta device (no weights, no storage, ~0.1s)
    and walk its Attention modules.

    The model is built under the SIMULATED tensor-parallel size (from the
    simulator scheduler config) so every spec reflects the per-rank view
    of the deployment: kv heads, mamba state shapes, and page sizes are
    all sharded exactly as a tp=N rank would see them.

    Requires the distributed environment to be initialized (the worker
    hook does this in init_device before building the stub).  Returns
    None when the meta build fails for a model family or version —
    callers then fall back to _build_kv_cache_spec.
    """
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.v1.kv_cache_interface import AttentionSpec
    except ImportError:
        return None

    # Simulated TP: the engine itself runs at tp=1 (forced by the
    # EngineArgs hook), but the spec must describe one rank of the
    # simulated tp=N deployment to match the per-rank memory accounting
    # (profile_device_available_bytes / calc_kv_cache_cell_elems).
    # Read the raw config JSON: this runs inside _initialize_kv_caches,
    # BEFORE the Scheduler hook has populated ConfigManager, and the JSON
    # scheduler section is the authoritative source (set_scheduler_config
    # gives external values priority over the engine's forced tp=1).
    sim_tp = 1
    try:
        from vllm_simulator.simulation.manager import ConfigManager
        sim_tp = int(
            ConfigManager._get_raw_config()
            .get("scheduler", {})
            .get("tp_size", 1)
            or 1
        )
    except Exception:
        pass

    try:
        kv_cache_spec: dict = {}

        import vllm.distributed.parallel_state as ps
        from vllm.platforms import current_platform

        tp_group = ps.get_tp_group()
        orig_world = tp_group.world_size
        orig_cfg_tp = vllm_config.parallel_config.tensor_parallel_size
        orig_oot = current_platform.is_out_of_tree
        orig_ep = getattr(ps, "_EP", None)

        # Scope-patches for the meta build only:
        # 1. Model code derives shard sizes from the RUNTIME TP group
        #    world size (read at call time via get_tp_group().world_size),
        #    not from parallel_config.  Meta device means no real
        #    communication ever happens on the mismatched group.
        # 2. The unquantized-MoE backend oracle assumes the platform is
        #    exactly one of cuda/rocm/xpu/cpu; the mock platform is
        #    deliberately none of them (is_cuda()=False skips CUDA-only
        #    import-time checks), which hits an UnboundLocalError.  Flag it
        #    out-of-tree — the designed escape hatch — so MoE models
        #    short-circuit to (OOT, None); experts_cls=None is only
        #    consumed in process_weights_after_loading, never run here.
        # 3. initialize_model_parallel creates the EP group only when the
        #    FIRST engine in the process is a MoE model, and later engines
        #    short-circuit on "already initialized" — so a MoE model built
        #    after a dense one finds _EP is None while MoE layer code calls
        #    get_ep_group() unconditionally.  Lend it the TP coordinator:
        #    MoE layers contribute no KV spec, and EP only sizes the meta
        #    expert weights, so the EP shape fidelity does not matter here.
        current_platform.is_out_of_tree = lambda: True
        if ps._EP is None:
            ps._EP = tp_group
        if sim_tp > 1:
            tp_group.world_size = sim_tp
            vllm_config.parallel_config.tensor_parallel_size = sim_tp
        try:
            with torch.device("meta"):
                initialize_model(vllm_config)
        finally:
            tp_group.world_size = orig_world
            vllm_config.parallel_config.tensor_parallel_size = orig_cfg_tp
            current_platform.is_out_of_tree = orig_oot
            ps._EP = orig_ep

        attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
        for layer_name, attn_module in attn_layers.items():
            if (
                isinstance(attn_module, Attention)
                and attn_module.kv_sharing_target_layer_name
            ):
                continue
            spec = attn_module.get_kv_cache_spec(vllm_config)
            if spec is None:
                continue
            if isinstance(spec, AttentionSpec):
                spec = attn_module.get_attn_backend().customize_spec(spec)
            kv_cache_spec[layer_name] = spec

        logger.info(
            "[vLLM Hijack] Built KV cache spec via meta-device model walk: "
            "%d layers (sim tp=%d)", len(kv_cache_spec), sim_tp,
        )
        return kv_cache_spec
    except Exception as e:
        import traceback

        logger.warning(
            "[vLLM Hijack] meta-device spec derivation failed "
            "(%s: %s); falling back to config-only derivation\n%s",
            type(e).__name__, e, traceback.format_exc(),
        )
        return None


def build_kv_cache_spec(vllm_config) -> dict:
    """Primary entry: native meta-device derivation, config-only fallback."""
    return _native_meta_spec(vllm_config) or _build_kv_cache_spec(vllm_config)


def _build_kv_cache_spec(vllm_config) -> dict:
    """Fallback: build KV cache spec from HF model config alone.

    Used only when the meta-device model walk is unavailable.  Known
    divergences from native (measured on Qwen3.5-9B): mamba layers use
    dummy shapes, bf16 state dtype, and page_size_padded forced to the
    attention page size instead of the natural ~2MB fp32 state page.

    Uses REAL num_kv_heads and head_size so that V6D object layout
    (page_size_bytes) matches the production server exactly — declared
    sizes need no scale compensation.  Physical allocation stays MINIMAL
    (1 page per tensor), so real page sizes cost no memory.

    Handles both pure-MHA models and hybrid models (e.g. Qwen3.5 with
    full_attention + linear_attention layers).
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    # MambaAttentionBackendEnum moved between vLLM versions
    try:
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        _mamba_gdn_type = MambaAttentionBackendEnum.GDN_ATTN
    except (ImportError, ModuleNotFoundError):
        try:
            from vllm.attention.backends.registry import MambaAttentionBackendEnum
            _mamba_gdn_type = MambaAttentionBackendEnum.GDN_ATTN
        except (ImportError, ModuleNotFoundError):
            _mamba_gdn_type = None

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    hf_config = model_config.hf_text_config

    # Real num_kv_heads (divided by TP for per-shard spec)
    scheduler_config = None
    try:
        scheduler_config = ConfigManager.get_scheduler_config()
    except Exception:
        pass

    tp_size = scheduler_config.tp_size if scheduler_config else 1
    # Support both real ModelConfig (has method) and SimpleNamespace mocks
    if hasattr(model_config, "get_total_num_kv_heads"):
        total_num_kv_heads = model_config.get_total_num_kv_heads()
    else:
        total_num_kv_heads = getattr(hf_config, "num_key_value_heads",
                                     hf_config.num_attention_heads)
    num_kv_heads = max(total_num_kv_heads // tp_size, 1)

    # Real head_size from the model config (page sizes are accounting-only)
    if hasattr(model_config, "get_head_size"):
        head_size = model_config.get_head_size()
    else:
        head_size = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )

    block_size = cache_config.block_size
    dtype = model_config.dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    # Determine layer types
    layer_types = getattr(hf_config, "layer_types", None)
    num_hidden_layers = hf_config.num_hidden_layers

    # Determine KV cache dtype ("auto" keeps the model dtype)
    cache_dtype = getattr(cache_config, "cache_dtype", "auto")
    if cache_dtype == "fp8":
        dtype = torch.float8_e4m3fn

    # Detect FullAttentionSpec supported fields (varies across vLLM versions)
    _fa_fields = {f.name for f in dataclasses.fields(FullAttentionSpec)}
    _fa_kwargs = dict(num_kv_heads=num_kv_heads, head_size=head_size, dtype=dtype)
    if "use_mla" in _fa_fields:
        _fa_kwargs["use_mla"] = False
    if "block_size" in _fa_fields:
        _fa_kwargs["block_size"] = block_size

    full_attn_spec = FullAttentionSpec(**_fa_kwargs)
    attn_page_size = full_attn_spec.page_size_bytes

    if layer_types is None:
        # Pure MHA model - use layer name format matching vLLM convention
        kv_cache_spec: dict = {
            f"model.layers.{i}": full_attn_spec
            for i in range(num_hidden_layers)
        }
    else:
        # Hybrid model: build per-layer specs with uniform page size.
        mamba_block_size = getattr(cache_config, "mamba_block_size", None)
        if mamba_block_size is None:
            mamba_block_size = block_size

        # Build MambaSpec
        mamba_kwargs = dict(
            block_size=mamba_block_size,
            shapes=((1, 1), (1, 1, 1)),
            dtypes=(dtype, dtype),
            page_size_padded=attn_page_size,
            mamba_cache_mode=getattr(cache_config, "mamba_cache_mode", "none"),
        )
        # Forward num_speculative_blocks (MTP/EAGLE) so light-mode runtime
        # block accounting matches the real deployment: the fork's
        # MambaManager uses _num_runtime_blocks = 1 + num_speculative_blocks
        # per request.  Leaving it at the default 0 makes each request hold
        # 3 fewer blocks than real (MTP k=3), underestimating pool eviction
        # pressure and letting cached mamba state snapshots live too long
        # (task29: prefix-cache hit ratio overestimated by up to +10.3pp on
        # node1_0047).
        _mamba_fields = {f.name for f in dataclasses.fields(MambaSpec)}
        if "num_speculative_blocks" in _mamba_fields:
            spec_cfg = getattr(vllm_config, "speculative_config", None)
            num_spec_tokens = (
                getattr(spec_cfg, "num_speculative_tokens", 0) or 0
            ) if spec_cfg is not None else 0
            mamba_kwargs["num_speculative_blocks"] = num_spec_tokens
        mamba_type_field = next(
            (f for f in dataclasses.fields(MambaSpec) if f.name == "mamba_type"),
            None,
        )
        if mamba_type_field is not None:
            if mamba_type_field.type == str or mamba_type_field.type == "str":
                mamba_kwargs["mamba_type"] = "gdn_attention"
            elif _mamba_gdn_type is not None:
                mamba_kwargs["mamba_type"] = _mamba_gdn_type

        mamba_spec = MambaSpec(**mamba_kwargs)

        kv_cache_spec = {}
        for i, layer_type in enumerate(layer_types):
            layer_name = f"model.layers.{i}"
            if layer_type == "full_attention":
                kv_cache_spec[layer_name] = full_attn_spec
            elif layer_type == "linear_attention":
                kv_cache_spec[layer_name] = mamba_spec
            else:
                kv_cache_spec[layer_name] = full_attn_spec

    # Speculative draft model layers: the real server's get_kv_cache_spec()
    # walks static_forward_context, which also contains the draft model's
    # attention layers — qwen3_5_mtp registers mtp_num_hidden_layers
    # full-attention layers (same head config as the main model) under
    # model.mtp.layers.<num_hidden_layers + idx>.  Mirror that here so the
    # v6d attention group has the same layer count; otherwise every block
    # object is under-sized by mtp_num_hidden_layers pages (real server:
    # 16 attn layers = 132 MiB/block, sim was 15 = 123.75 MiB/block).
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    spec_method = getattr(spec_cfg, "method", None) if spec_cfg else None
    num_mtp_layers = (
        getattr(hf_config, "mtp_num_hidden_layers", 0) or 0
        if spec_method not in (None, "ngram", "suffix")
        else 0
    )
    for idx in range(num_mtp_layers):
        layer_name = f"model.mtp.layers.{num_hidden_layers + idx}"
        kv_cache_spec[layer_name] = full_attn_spec

    logger.info(
        "[V6D Hijack] Built KV cache spec: %d layers (incl. %d MTP draft), "
        "num_kv_heads=%d (total=%d, tp=%d), head_size=%d, block_size=%d, "
        "page_size=%d bytes",
        len(kv_cache_spec), num_mtp_layers, num_kv_heads, total_num_kv_heads,
        tp_size, head_size, block_size, attn_page_size,
    )
    return kv_cache_spec
