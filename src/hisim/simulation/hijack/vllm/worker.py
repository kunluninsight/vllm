"""Simulate vLLM worker execution without constructing a GPU model runner.

Build KV specs from model dimensions for native cache accounting, but allocate
at most 4 KiB per CPU backing tensor. v6d object sizes follow the specs, not
these buffers. Mock token output retains the connector lifecycle.
"""

import asyncio
import dataclasses
import threading
import time

import torch

from hisim.hook import BaseHook
from hisim.simulation.manager import ConfigManager
from hisim.simulation.utils import profile_device_available_bytes
from hisim.simulation.hijack.vllm.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from hisim.utils import get_logger

logger = get_logger()


class _ModelRunnerStub:
    """Picklable model-runner placeholder with cached KV specs."""

    def __init__(self, kv_spec=None):
        self._kv_spec = kv_spec
        self.model_memory_usage = 0

    def get_kv_cache_spec(self):
        return self._kv_spec

    def __getattr__(self, name):
        # Public fallback methods are no-ops; private attributes stay absent.
        if name.startswith("_"):
            raise AttributeError(name)
        return _noop


def _noop(*args, **kwargs):
    """Module-level no-op function (picklable)."""
    return None


def _build_kv_cache_spec(vllm_config) -> dict:
    """Build attention/Mamba accounting specs without allocating their payloads."""
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

    # Use target TP for per-shard KV heads when scheduler config is available.
    scheduler_config = None
    try:
        scheduler_config = ConfigManager.get_scheduler_config()
    except Exception:
        pass

    tp_size = scheduler_config.tp_size if scheduler_config else 1
    if hasattr(model_config, "get_total_num_kv_heads"):
        total_num_kv_heads = model_config.get_total_num_kv_heads()
    else:
        total_num_kv_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        )
    num_kv_heads = max(total_num_kv_heads // tp_size, 1)

    if hasattr(model_config, "get_head_size"):
        head_size = model_config.get_head_size()
    else:
        head_size = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )

    block_size = cache_config.block_size
    # Read the optional physical size before scheduler config initialization;
    # update cache_config so scheduling and KV specs use the same block size.
    try:
        _injected_block_size = (
            ConfigManager._get_raw_config()
            .get("scheduler", {})
            .get("physical_block_size")
        )
    except Exception:
        _injected_block_size = None
    if _injected_block_size:
        block_size = int(_injected_block_size)
        cache_config.block_size = block_size
        if getattr(cache_config, "mamba_block_size", None) is not None:
            cache_config.mamba_block_size = block_size
    # Mamba padding must include the merged attention pack's page growth.
    attn_pack_size = getattr(cache_config, "attn_pack_size", 1) or 1
    dtype = model_config.dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

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

    if not layer_types or "linear_attention" not in layer_types:
        # Explicit attention-only layer metadata does not require a Mamba spec.
        kv_cache_spec: dict = {
            f"model.layers.{i}": full_attn_spec for i in range(num_hidden_layers)
        }
    else:
        # Hybrid model: build per-layer specs with uniform page size.
        mamba_block_size = getattr(cache_config, "mamba_block_size", None)
        if mamba_block_size is None or _injected_block_size:
            mamba_block_size = block_size

        # Resolve model-specific state shapes/dtypes; retain placeholders on failure.
        mamba_shapes = ((1, 1), (1, 1, 1))
        mamba_dtypes = (dtype, dtype)
        try:
            try:
                from vllm.model_executor.models import ModelRegistry
            except ImportError:
                from vllm.model_executor.models.registry import ModelRegistry
            model_cls, _ = ModelRegistry.resolve_model_cls(
                model_config.architecture, model_config=model_config
            )
            mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
            mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
        except Exception:
            logger.warning(
                "[V6D Hijack] failed to resolve real mamba state spec, "
                "falling back to placeholder shapes/dtypes",
                exc_info=True,
            )

        # Match the padded Mamba page to the merged attention page.
        mamba_kwargs = dict(
            block_size=mamba_block_size,
            shapes=mamba_shapes,
            dtypes=mamba_dtypes,
            page_size_padded=attn_page_size * attn_pack_size,
            mamba_cache_mode=getattr(cache_config, "mamba_cache_mode", "none"),
        )
        # Speculative blocks contribute to per-request Mamba cache pressure.
        _mamba_fields = {f.name for f in dataclasses.fields(MambaSpec)}
        if "state_align_size" in _mamba_fields:
            # light_flex uses 64-token state boundaries; other modes use alignment 1.
            mamba_kwargs["state_align_size"] = (
                64 if mamba_kwargs["mamba_cache_mode"] == "light_flex" else 1
            )
        if "num_speculative_blocks" in _mamba_fields:
            spec_cfg = getattr(vllm_config, "speculative_config", None)
            num_spec_tokens = (
                (getattr(spec_cfg, "num_speculative_tokens", 0) or 0)
                if spec_cfg is not None
                else 0
            )
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

    # Include configured MTP attention layers in declared KV object sizes.
    # Lookup-only speculative methods do not add draft-model layers.
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
        "page_size=%d bytes, mamba_page_padded=%d bytes, attn_pack_size=%d",
        len(kv_cache_spec),
        num_mtp_layers,
        num_kv_heads,
        total_num_kv_heads,
        tp_size,
        head_size,
        block_size,
        attn_page_size,
        attn_page_size * attn_pack_size,
        attn_pack_size,
    )
    return kv_cache_spec


class C_VLLMWorkerHook(BaseHook):
    """Replace GPU execution with CPU placeholders and native connector calls."""

    HOOK_CLASS_NAME = "Worker"
    HOOK_MODULE_NAME = r"vllm\.v1\.worker\.(gpu_worker|worker)"
    REGEX = True

    @classmethod
    def hook(cls, target):
        from vllm.v1.outputs import (
            AsyncModelRunnerOutput as _AsyncModelRunnerOutput,
            ModelRunnerOutput as _ModelRunnerOutput,
        )

        class _SimAsyncOutput(_AsyncModelRunnerOutput):
            """Defer a simulated GPU span to vLLM's async-output thread."""

            def __init__(self, output, latency: float):
                self._output = output
                self._latency = latency

            def get_output(self):
                time.sleep(self._latency)
                return self._output

        # Field set varies across vLLM versions; probe once at install time.
        _mro_fields = {f.name for f in dataclasses.fields(_ModelRunnerOutput)}

        _KVConnectorOutput = None
        try:
            from vllm.v1.outputs import KVConnectorOutput as _KVConnectorOutput
        except ImportError:
            pass

        _has_kv_transfer_group = None
        _get_kv_transfer_group = None
        try:
            from vllm.distributed.kv_transfer import (
                has_kv_transfer_group as _has_kv_transfer_group,
                get_kv_transfer_group as _get_kv_transfer_group,
            )
        except ImportError:
            pass

        def override_init_device(self):
            """Initialize Gloo, CPU workspace, and the KV-spec runner stub."""
            try:
                from vllm.v1.worker.gpu_worker import (
                    init_worker_distributed_environment,
                    init_workspace_manager,
                    set_random_seed,
                )
            except ImportError:
                from vllm.v1.worker.worker import (
                    init_worker_distributed_environment,
                    init_workspace_manager,
                    set_random_seed,
                )

            # Supply the config context required by the distributed initializer.
            try:
                from vllm.config.vllm import set_current_vllm_config

                with set_current_vllm_config(self.vllm_config):
                    init_worker_distributed_environment(
                        self.vllm_config,
                        self.rank,
                        self.distributed_init_method,
                        self.local_rank,
                        "gloo",
                    )
            except (ImportError, TypeError):
                # Compatibility path without the config context helper.
                init_worker_distributed_environment(
                    self.vllm_config,
                    self.rank,
                    self.distributed_init_method,
                    self.local_rank,
                    "gloo",
                )
            set_random_seed(self.vllm_config.model_config.seed)

            self.device = torch.device("cpu")

            kv_spec = _build_kv_cache_spec(self.vllm_config)
            self.model_runner = _ModelRunnerStub(kv_spec=kv_spec)

            self.init_snapshot = None
            self.requested_memory = 0
            init_workspace_manager(self.device, 1)
            logger.info("[vLLM Hijack] Worker.init_device: stub initialized")

        def override_load_model(self, *args, **kwargs):
            """Skip model loading entirely."""
            logger.info("[vLLM Hijack] Worker.load_model: skipped")

        def override_determine_available_memory(self):
            """Profile modeled KV capacity, with an 80 GiB fallback on failure."""
            try:
                model = resolve_model_info(self.vllm_config.model_config)
                ConfigManager.set_model_info(model)
                hw = ConfigManager.get_accelerator_info()
                sched_config = resolve_scheduler_config(self.vllm_config)
                ConfigManager.set_scheduler_config(sched_config)
                available_bytes = profile_device_available_bytes(
                    model, hw, sched_config
                )

                logger.info(
                    "[vLLM Hijack] Worker.determine_available_memory: %d gibibytes. The available memory will be used for kv cache allocation.",
                    available_bytes // (1 << 30),
                )

                return available_bytes
            except Exception as e:
                # Profiling needs the optional aiconfigurator extra; without it
                # the fallback silently distorts the KV cache block count.
                logger.warning(
                    "[vLLM Hijack] Worker.determine_available_memory: profiling "
                    "failed (%s), falling back to 80 gibibytes. Install "
                    "hisim[aiconfigurator] for modeled KV capacity.",
                    e,
                )
                return 80 * (1 << 30)  # 80 GiB fallback

        def override_get_kv_cache_spec(self):
            """Return pre-built KV cache spec."""
            return self.model_runner.get_kv_cache_spec()

        def override_initialize_from_config(self, kv_cache_config):
            """Init KV connector and allocate CPU KV cache tensors."""
            from vllm.distributed.kv_transfer import (
                ensure_kv_transfer_initialized,
                has_kv_transfer_group,
                get_kv_transfer_group,
            )

            # Use the first shard when the engine supplies a config list.
            if isinstance(kv_cache_config, (list, tuple)):
                kv_cache_config = kv_cache_config[0]

            self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
            num_blocks = kv_cache_config.num_blocks

            # Provide the connector loop without GPU worker initialization.
            # Temporarily replace worker_init only during connector construction.
            try:
                import vllm.v1.hybrid_connector.engine_proxy as _engine_proxy

                _orig_worker_init = _engine_proxy.worker_init

                def _ensure_cpu_worker_loop():
                    if getattr(_engine_proxy, "_g_worker_loop", None) is not None:
                        return
                    loop = asyncio.new_event_loop()

                    def _run_loop():
                        asyncio.set_event_loop(loop)
                        loop.run_forever()

                    thread = threading.Thread(
                        target=_run_loop,
                        name="hybridworker-cpu-loop",
                        daemon=True,
                    )
                    thread.start()
                    _engine_proxy._g_worker_loop = loop
                    logger.info("[vLLM Hijack] installed CPU hybrid worker loop")

                def _cpu_worker_init(vllm_config, local_rank):
                    _ensure_cpu_worker_loop()
                    return None

                _ensure_cpu_worker_loop()
                _engine_proxy.worker_init = _cpu_worker_init
            except (ImportError, AttributeError):
                _orig_worker_init = None

            try:
                ensure_kv_transfer_initialized(
                    self.vllm_config, kv_cache_config, self.local_rank
                )
            except TypeError:
                # Compatibility signature without local_rank.
                ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
            finally:
                if _orig_worker_init is not None:
                    _engine_proxy.worker_init = _orig_worker_init

            # Keep native block counts, but cap CPU payload buffers at 4 KiB.
            kv_caches: dict = {}
            if (
                hasattr(kv_cache_config, "kv_cache_tensors")
                and kv_cache_config.kv_cache_tensors
            ):
                for kv_tensor in kv_cache_config.kv_cache_tensors:
                    minimal_size = min(kv_tensor.size, 4096)
                    tensor = torch.zeros(minimal_size, dtype=torch.int8, device="cpu")
                    for layer_name in kv_tensor.shared_by:
                        kv_caches[layer_name] = tensor
                logger.info(
                    "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
                    "(num_blocks=%d, simulated_bytes=%d, actual_bytes=4k)",
                    len(kv_cache_config.kv_cache_tensors),
                    num_blocks,
                    sum(t.size for t in kv_cache_config.kv_cache_tensors),
                )
            else:
                kv_spec = self.model_runner.get_kv_cache_spec()
                for layer_name, spec in kv_spec.items():
                    minimal_size = min(spec.page_size_bytes, 4096)
                    tensor = torch.zeros(minimal_size, dtype=torch.int8, device="cpu")
                    kv_caches[layer_name] = tensor
                logger.info(
                    "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
                    "(num_blocks=%d, page_size=%d, actual_alloc=4k)",
                    len(kv_caches),
                    num_blocks,
                    spec.page_size_bytes if kv_spec else 0,
                )

            if has_kv_transfer_group():
                kv_transfer_group = get_kv_transfer_group()
                kv_transfer_group.register_kv_caches(kv_caches)
                logger.info("[vLLM Hijack] KV caches registered with connector")

            logger.info(
                "[vLLM Hijack] Worker.initialize_from_config: "
                "V6D-aware init complete, num_blocks=%d",
                num_blocks,
            )

        def override_compile_or_warm_up_model(self):
            """Skip compilation and warmup entirely."""
            logger.info("[vLLM Hijack] Worker.compile_or_warm_up_model: skipped")
            try:
                from vllm.v1.worker.worker_base import CompilationTimes

                return CompilationTimes(language_model=0.0, encoder=0.0)
            except (ImportError, ModuleNotFoundError, TypeError):
                return None

        def override_get_supported_tasks(self):
            """Return generate task (default for causal LM)."""
            return ("generate",)

        def override_execute_model(self, scheduler_output):
            """Return mock ModelRunnerOutput with KV connector lifecycle."""
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens

            # KV connector pre-forward
            kv_connector = None
            try:
                if _has_kv_transfer_group and _has_kv_transfer_group():
                    kv_connector = _get_kv_transfer_group()
                    kv_connector_metadata = scheduler_output.kv_connector_metadata
                    logger.debug(
                        "[V6D Hijack] execute_model: kv_connector=%s metadata=%s",
                        type(kv_connector).__name__,
                        type(kv_connector_metadata).__name__
                        if kv_connector_metadata is not None
                        else None,
                    )
                    if kv_connector_metadata is not None:
                        if hasattr(kv_connector, "handle_preemptions"):
                            kv_connector.handle_preemptions(kv_connector_metadata)
                        kv_connector.bind_connector_metadata(kv_connector_metadata)
                        kv_connector.start_load_kv(None)
                else:
                    logger.debug("[V6D Hijack] execute_model: no kv_transfer_group")
            except Exception:
                # Retain the connector so post-forward cleanup can still run.
                # A pre-forward failure must not suppress completion harvesting.
                logger.critical(
                    "[V6D Hijack] execute_model: KV pre-forward lifecycle "
                    "failed (engine metadata contract changed?); continuing "
                    "with the post-forward harvest so pending store/load "
                    "deadlines keep draining",
                    exc_info=True,
                )

            req_ids = list(num_scheduled_tokens.keys()) if num_scheduled_tokens else []
            # Executor annotations suppress tokens for incomplete prefill chunks;
            # emitting early would finish short-output requests before full prefill.
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
            if "kv_lens" in _mro_fields:
                mro_kwargs["kv_lens"] = [0] * len(req_ids)
            if "pooler_output" in _mro_fields:
                mro_kwargs["pooler_output"] = [None] * len(req_ids)
            output = _ModelRunnerOutput(**mro_kwargs)

            # KV connector post-forward
            if kv_connector is not None and _KVConnectorOutput is not None:
                try:
                    kv_connector.wait_for_save()
                    kv_output = _KVConnectorOutput()
                    finished_req_ids = getattr(
                        scheduler_output, "finished_req_ids", set()
                    )
                    kv_output.finished_sending, kv_output.finished_recving = (
                        kv_connector.get_finished(finished_req_ids)
                    )
                    if kv_output.finished_sending or kv_output.finished_recving:
                        logger.info(
                            "[V6D Hijack] execute_model: finished_sending=%s "
                            "finished_recving=%s",
                            sorted(kv_output.finished_sending or []),
                            sorted(kv_output.finished_recving or []),
                        )
                    else:
                        logger.debug(
                            "[V6D Hijack] execute_model: finished_sending=[] "
                            "finished_recving=[]"
                        )
                    if hasattr(kv_connector, "build_connector_worker_meta"):
                        kv_output.kv_connector_worker_meta = (
                            kv_connector.build_connector_worker_meta()
                        )
                    kv_connector.clear_connector_metadata()
                    output.kv_connector_output = kv_output
                except Exception:
                    logger.exception(
                        "[V6D Hijack] execute_model: KV post-forward lifecycle failed"
                    )

            self._last_model_output = output
            latency = getattr(scheduler_output, "_sim_full_step_latency", None)
            scheduler_config = getattr(
                getattr(self, "vllm_config", None), "scheduler_config", None
            )
            async_scheduling = bool(
                getattr(scheduler_config, "async_scheduling", False)
            )
            self._last_sim_full_step_latency = latency if async_scheduling else None
            if latency is not None and not async_scheduling:
                # Honor an explicitly attached span even without async scheduling.
                return _SimAsyncOutput(output, latency)
            return output

        def override_sample_tokens(self, grammar_output):
            """Defer BLOCKING GPU time when async scheduling is enabled."""
            output = self._last_model_output
            latency = getattr(self, "_last_sim_full_step_latency", None)
            self._last_sim_full_step_latency = None
            if latency is not None:
                return _SimAsyncOutput(output, latency)
            return output

        def override_take_draft_token_ids(self):
            """Skip draft-token updates; speculative acceleration is not simulated."""
            return None

        def override_get_attn_backends_type(self):
            """Return empty list - no real attention backends in simulation."""
            return []

        def override_sleep(self, level=1):
            pass

        def override_wake_up(self, tags=None):
            pass

        def override_reset_mm_cache(self):
            """No-op: no real model_runner to reset."""
            pass

        def override_initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
            """Store block counts only."""
            self.cache_config.num_gpu_blocks = num_gpu_blocks
            self.cache_config.num_cpu_blocks = num_cpu_blocks

        def override_initialize_kv_transfer(self):
            """No-op in simulation."""
            pass

        target.init_device = override_init_device
        target.load_model = override_load_model
        target.determine_available_memory = override_determine_available_memory
        target.get_kv_cache_spec = override_get_kv_cache_spec
        target.get_supported_tasks = override_get_supported_tasks
        target.initialize_from_config = override_initialize_from_config
        target.compile_or_warm_up_model = override_compile_or_warm_up_model
        target.execute_model = override_execute_model
        target.take_draft_token_ids = override_take_draft_token_ids
        target.sample_tokens = override_sample_tokens
        target.sleep = override_sleep
        target.wake_up = override_wake_up
        target.get_attn_backends_type = override_get_attn_backends_type
        target.reset_mm_cache = override_reset_mm_cache
        target.initialize_cache = override_initialize_cache
        target.initialize_kv_transfer = override_initialize_kv_transfer
