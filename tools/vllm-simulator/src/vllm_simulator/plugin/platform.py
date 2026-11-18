# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution selected through vLLM's out-of-tree platform interface."""

import math
import sys
from copy import copy, deepcopy

from vllm.exceptions import VLLMValidationError
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.plugin.device import SimulatorDevice
from vllm_simulator.plugin.state import ReplayMetadata


class SimulatorPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_type = "cpu"
    device_name = "simulator"
    dispatch_key = "CPU"

    _device: SimulatorDevice | None = None

    @classmethod
    def _get_device(cls) -> SimulatorDevice:
        # Configuration imports can query the platform before VllmConfig exists.
        get_config = getattr(
            sys.modules.get("vllm.config.vllm"), "get_current_vllm_config_or_none", None
        )
        config = get_config() if get_config is not None else None
        if config is not None:
            options = config.additional_config.get("simulator", {})
            return cls._resolve_device(options.get("device"))
        if cls._device is None:
            cls._device = cls._resolve_device(None)
        return cls._device

    @classmethod
    def configure_device(cls, value) -> SimulatorDevice:
        cls._device = cls._resolve_device(value)
        return cls._device

    @classmethod
    def get_device_name(cls, device_id=0):
        return cls._get_device().name

    @classmethod
    def get_device_capability(cls, device_id=0):
        return DeviceCapability(*cls._get_device().compute_capability)

    @classmethod
    def get_device_total_memory(cls, device_id=0):
        return cls._get_device().total_memory_bytes

    @classmethod
    def num_compute_units(cls, device_id=0):
        return cls._get_device().num_compute_units

    @classmethod
    def support_hybrid_kv_cache(cls):
        return True

    @classmethod
    def import_kernels(cls) -> None:
        pass

    @classmethod
    def _resolve_device(cls, value):
        if isinstance(value, SimulatorDevice):
            return value
        default = {
            "name": "dummy",
            "total_memory_gb": 288,
            "compute_capability": (10, 3),
            "num_compute_units": 148,
        }
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("simulator.device must be a JSON object")
            default.update(value)
        return SimulatorDevice(**default)

    @classmethod
    def apply_config_platform_defaults(cls, config) -> None:
        from vllm.config.compilation import CompilationMode, CUDAGraphMode

        options = dict(config.additional_config.get("simulator", {}))
        options["device"] = cls.configure_device(options.get("device"))
        config.additional_config = {**config.additional_config, "simulator": options}
        if config.model_config is None:
            return
        from vllm_simulator.plugin.core import install_simulator_launcher

        simulator_config = SimulatorConfig.backup_parallel_config(config)
        parallel = config.parallel_config
        parallel.tensor_parallel_size = 1
        parallel.pipeline_parallel_size = 1
        parallel.prefill_context_parallel_size = 1
        parallel.decode_context_parallel_size = 1
        parallel.enable_expert_parallel = False
        parallel.enable_eplb = False
        parallel.world_size = 1
        parallel.rank = 0
        parallel.nnodes = 1
        parallel.node_rank = 0
        parallel.master_addr = "127.0.0.1"
        if not parallel.data_parallel_external_lb:
            parallel.data_parallel_size_local = parallel.data_parallel_size
            parallel.data_parallel_rank = 0
            parallel.data_parallel_rank_local = None
            parallel.data_parallel_index = 0
            parallel.data_parallel_master_ip = "127.0.0.1"
        cls._adapt_kv_transfer(config)
        config.parallel_config.worker_cls = (
            "vllm_simulator.plugin.worker.SimulatorWorker"
        )
        config.parallel_config.distributed_executor_backend = (
            "vllm_simulator.plugin.executor.SimulatorExecutor"
        )
        config._post_init_kv_transfer_config()
        topology = simulator_config.original_parallel_config
        requested_async = config.scheduler_config.async_scheduling
        use_async = (
            requested_async is not False
            and topology.data_parallel_size == topology.pipeline_parallel_size == 1
            and config.kv_transfer_config is None
        )
        config.scheduler_config.scheduler_cls = (
            "vllm_simulator.plugin.scheduler.SimulatorAsyncScheduler"
            if use_async
            else "vllm_simulator.plugin.scheduler.SimulatorScheduler"
        )
        config.scheduler_config.async_scheduling = use_async
        config.cache_config.enable_extensible_kv_cache = False
        config.cache_config.block_size = (
            simulator_config.physical_kv_cache_block_size
            * simulator_config.original_parallel_config.decode_context_parallel_size
        )
        config.model_config.enforce_eager = True
        config.compilation_config.mode = CompilationMode.NONE
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        install_simulator_launcher()

    @classmethod
    def _adapt_kv_transfer(cls, config) -> None:
        from vllm_simulator.plugin.kv_transfer.protocol import (
            CONNECTOR_MODULE,
            CONNECTOR_NAME,
        )

        simulation = SimulatorConfig.from_config(config)
        transfer = config.kv_transfer_config
        if transfer is None:
            if simulation.kv_transfer:
                raise ValueError("simulator.kv_transfer requires --kv-transfer-config")
            return
        native = transfer.kv_connector in ("NixlConnector", "MooncakeConnector")
        if not native and (
            transfer.kv_connector != CONNECTOR_NAME
            or transfer.kv_connector_module_path != CONNECTOR_MODULE
        ):
            if simulation.kv_transfer:
                raise ValueError(
                    "simulator.kv_transfer requires a supported P/D connector"
                )
            return
        if native:
            if transfer.kv_connector_module_path is not None:
                raise ValueError("Simulator cannot adapt custom connector modules")
            extra = transfer.kv_connector_extra_config
            transport_options = (
                {"backends", "ucx_backend_params"}
                if transfer.kv_connector == "NixlConnector"
                else {"mooncake_protocol", "device_name", "bootstrap_server_address"}
            )
            if extra.keys() - transport_options:
                raise ValueError(
                    "Unsupported native connector settings for simulation: "
                    f"{sorted(extra.keys() - transport_options)}"
                )
            if transfer.kv_connector == "NixlConnector" and extra.get(
                "backends", ["UCX"]
            ) != ["UCX"]:
                raise ValueError("Simulator NixlConnector adaptation supports UCX only")
            config.additional_config["simulator"].setdefault(
                "original_kv_transfer_config", deepcopy(transfer)
            )
        adapted = copy(transfer)
        role = simulation.kv_transfer.get("kv_role", transfer.kv_role)
        if transfer.kv_role != "kv_both" and role != transfer.kv_role:
            raise ValueError(
                "simulator.kv_transfer.kv_role conflicts with the native role"
            )
        if role not in ("kv_producer", "kv_consumer"):
            raise ValueError(
                "P/D simulation requires simulator.kv_transfer.kv_role "
                "to select kv_producer or kv_consumer for kv_both"
            )
        adapted.kv_connector = CONNECTOR_NAME
        adapted.kv_connector_module_path = CONNECTOR_MODULE
        adapted.kv_buffer_device = "cpu"
        adapted.kv_role = role
        adapted.kv_connector_extra_config = {
            **({} if native else transfer.kv_connector_extra_config),
            **{
                key: value
                for key, value in simulation.kv_transfer.items()
                if key != "kv_role"
            },
        }
        config.kv_transfer_config = adapted

    @classmethod
    def check_and_update_config(cls, config) -> None:
        if config.model_config is None:
            return
        simulator_config = SimulatorConfig.from_config(config)
        parallel = config.parallel_config
        if parallel.data_parallel_backend != "mp":
            raise ValueError("Simulator DP currently requires the mp backend")
        if parallel.data_parallel_hybrid_lb:
            raise ValueError("Simulator DP does not support hybrid load balancing")
        if parallel.data_parallel_external_lb and simulator_config.mode != "BLOCKING":
            raise ValueError("Simulator external load balancing requires BLOCKING mode")
        if (
            parallel.data_parallel_size > 1
            and not config.model_config.is_moe
            and simulator_config.mode == "NON_BLOCKING"
        ):
            raise ValueError(
                "NON_BLOCKING replay requires a shared clock across DP ranks. "
                "Dense DP ranks run independently. Use BLOCKING mode for dense DP "
                "or data_parallel_size=1 for virtual-time replay."
            )
        if parallel.enable_dbo or parallel.enable_elastic_ep:
            raise ValueError("Simulator DP does not support DBO or elastic EP")
        model = config.model_config
        if (
            model.is_encoder_decoder
            or model.is_attention_free
            or model.get_num_layers(config.parallel_config) == 0
        ):
            raise ValueError("The simulator requires a decoder with KV cache layers")
        if (
            config.speculative_config
            or config.lora_config
            or config.ec_transfer_config
            or model.runner_type != "generate"
        ):
            raise ValueError(
                "Speculation, LoRA, encoder connectors and pooling are not supported"
            )
        cls._configure_offload(config)
        if simulator_config.token_id >= model.get_vocab_size():
            raise ValueError("simulator.token_id must be inside the model vocabulary")

    @classmethod
    def _configure_offload(cls, config) -> None:
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            simple_cpu_offload_connector,
        )

        transfer = config.kv_transfer_config
        if transfer is None:
            return
        if (
            transfer.kv_connector == "SimulatorOfflineKVConnector"
            and transfer.kv_connector_module_path
            == "vllm_simulator.plugin.kv_transfer.offline"
        ):
            simulation = SimulatorConfig.from_config(config)
            original = simulation.original_parallel_config
            if (
                simulation.mode != "NON_BLOCKING"
                or transfer.kv_role != "kv_consumer"
                or any(
                    getattr(original, key) != 1
                    for key in (
                        "data_parallel_size",
                        "pipeline_parallel_size",
                        "prefill_context_parallel_size",
                        "decode_context_parallel_size",
                    )
                )
            ):
                raise ValueError(
                    "Offline KV import requires NON_BLOCKING, kv_consumer "
                    "and DP=PP=PCP=DCP=1"
                )
            return
        from vllm_simulator.plugin.kv_transfer.protocol import (
            CONNECTOR_MODULE,
            CONNECTOR_NAME,
        )

        if (
            transfer.kv_connector == CONNECTOR_NAME
            and transfer.kv_connector_module_path == CONNECTOR_MODULE
        ):
            simulation = SimulatorConfig.from_config(config)
            if (
                simulation.mode != "BLOCKING"
                or config.parallel_config.data_parallel_size != 1
                or transfer.kv_role not in ("kv_producer", "kv_consumer")
                or any(
                    getattr(simulation.original_parallel_config, name) != 1
                    for name in (
                        "pipeline_parallel_size",
                        "prefill_context_parallel_size",
                        "decode_context_parallel_size",
                    )
                )
            ):
                raise ValueError(
                    "Simulator PD requires BLOCKING, DP=PP=PCP=DCP=1 and P/D roles"
                )
            from urllib.parse import urlparse

            extra = transfer.kv_connector_extra_config
            url = urlparse(extra.get("broker_url", ""))
            if url.scheme not in ("http", "https") or not url.hostname:
                raise ValueError("Simulator PD requires an HTTP(S) broker_url")
            for key, default in (("rpc_timeout_s", 2), ("poll_interval_s", 0.005)):
                value = extra.get(key, default)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"Simulator PD {key} must be finite and positive")
            return
        if (
            transfer.kv_connector
            not in ("SimpleCPUOffloadConnector", "OffloadingConnector")
            or transfer.kv_connector_module_path is not None
            or transfer.kv_role != "kv_both"
        ):
            raise ValueError(
                "Simulator KV offload requires native SimpleCPUOffloadConnector or "
                "OffloadingConnector with kv_role=kv_both"
            )
        extra = dict(transfer.kv_connector_extra_config)
        if extra.get("kv_offload_backend", "cpu") != "cpu":
            raise ValueError("Simulator KV offload supports only the CPU backend")
        if transfer.kv_connector == "SimpleCPUOffloadConnector":
            original = SimulatorConfig.from_config(config).original_parallel_config
            extra.setdefault(
                "cpu_bytes_to_use_per_rank",
                int(
                    extra.get(
                        "cpu_bytes_to_use",
                        simple_cpu_offload_connector.DEFAULT_CPU_CAPACITY_BYTES,
                    )
                )
                // original.world_size,
            )
            capacity = extra["cpu_bytes_to_use_per_rank"]
        else:
            if (
                extra.get("spec_name", "CPUOffloadingSpec") != "CPUOffloadingSpec"
                or extra.get("spec_module_path") is not None
                or extra.get("canonical_layout", False)
                or extra.get("secondary_tiers")
            ):
                raise ValueError(
                    "Simulator OffloadingConnector requires CPUOffloadingSpec "
                    "with the native local layout and no secondary tiers"
                )
            capacity = extra.get("cpu_bytes_to_use", 0)
        if int(capacity) <= 0:
            raise ValueError("Simulator CPU offload capacity must be positive")
        transfer.kv_connector_extra_config = extra

    @classmethod
    def get_scheduler_config(cls, config, kv_cache_config):
        """Return a scheduler config view with a single-worker offload budget."""
        from vllm.utils.math_utils import round_up
        from vllm_simulator.plugin.kv_transfer.capacity import cpu_offloading_spec

        transfer = config.kv_transfer_config
        if transfer is None or transfer.kv_connector != "OffloadingConnector":
            return config
        spec = cpu_offloading_spec(config, kv_cache_config)
        view = copy(config)
        view.kv_transfer_config = copy(transfer)
        view.kv_transfer_config.kv_connector_extra_config = {
            **transfer.kv_connector_extra_config,
            "cpu_bytes_to_use": spec.num_chunks
            * round_up(spec.cpu_page_size_per_worker, spec.BLOCK_SIZE_ALIGNMENT),
        }
        if spec.num_chunks == 0:
            raise ValueError("Simulator CPU offload capacity is smaller than one chunk")
        return view

    @classmethod
    def validate_request(cls, processed_inputs, params) -> None:
        if any(processed_inputs.get("mm_placeholders", {}).values()):
            raise VLLMValidationError("The simulator supports text inputs only")
        if params.n != 1:
            raise VLLMValidationError("The simulator requires n=1")
        transfer = (params.extra_args or {}).get("kv_transfer_params") or {}
        if not isinstance(transfer, dict):
            raise VLLMValidationError("kv_transfer_params must be an object")
        if (
            transfer.get("do_remote_decode") or transfer.get("do_remote_prefill")
        ) and processed_inputs.get("prompt_embeds") is not None:
            raise VLLMValidationError("Simulator P/D handoff requires prompt token IDs")
        if "offline_kv_tokens" in transfer:
            from vllm_simulator.plugin.kv_transfer.offline import validate_import

            try:
                validate_import(
                    transfer, len(processed_inputs.get("prompt_token_ids") or [])
                )
            except ValueError as error:
                raise VLLMValidationError(str(error)) from error
            if params.skip_reading_prefix_cache:
                raise VLLMValidationError(
                    "Offline KV import requires external KV reads"
                )
        if transfer.get("do_remote_decode") and params.max_tokens != 1:
            raise VLLMValidationError("Simulator P handoff requires max_tokens=1")
        if transfer.get("do_remote_prefill") and not all(
            isinstance(transfer.get(key), str) and transfer[key]
            for key in ("remote_engine_id", "remote_request_id")
        ):
            raise VLLMValidationError(
                "Simulator D requires the P response's kv_transfer_params"
            )
        if transfer.get("do_remote_prefill") and params.skip_reading_prefix_cache:
            raise VLLMValidationError("Simulator D requires external KV reads")
        try:
            ReplayMetadata.from_extra_args(params.extra_args, 0.0)
        except (ValueError, TypeError) as error:
            raise VLLMValidationError(str(error)) from error
        if params.logprobs is not None or params.prompt_logprobs is not None:
            raise VLLMValidationError(
                "The simulator does not compute log probabilities"
            )
        if params.structured_outputs is not None:
            raise VLLMValidationError(
                "The simulator does not implement constrained sampling"
            )
