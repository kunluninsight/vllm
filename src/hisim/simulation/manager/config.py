from typing import Optional

from hisim.env import Envs
from hisim.simulation.types import PlatformConfig, SchedulerConfig
from hisim.spec import AcceleratorInfo, DataType, ModelInfo
from hisim.time_predictor import InferTimePredictor, LinearTimePredictor
from hisim.utils import get_logger

logger = get_logger()


class ConfigManager:
    """Centralized configuration manager with caching."""

    _model_info: Optional[ModelInfo] = None
    _platform_config: Optional[PlatformConfig] = None
    _scheduler_config: Optional[SchedulerConfig] = None
    _raw_config: Optional[dict] = None

    @classmethod
    def _get_raw_config(cls) -> dict:
        if cls._raw_config is None:
            cls._raw_config = Envs.load_config()
        return cls._raw_config

    @classmethod
    def reset_config_cache(cls):
        cls._raw_config = None
        cls._model_info = None
        cls._platform_config = None
        cls._scheduler_config = None

    @classmethod
    def set_model_info(cls, model: ModelInfo):
        cls._model_info = model

    @classmethod
    def get_model_info(cls) -> ModelInfo | None:
        return cls._model_info

    @classmethod
    def get_accelerator_info(cls) -> AcceleratorInfo:
        config = cls._get_raw_config()
        platform_config = config.get("platform", {})
        acc_info = platform_config.get("accelerator", {})
        hw = AcceleratorInfo.find_by_hw_name(acc_info.get("name"))
        if hw is None:
            logger.debug(
                f"Failed to initialize device info with {acc_info.get('name')}. All available devices are: {AcceleratorInfo.list_all_hws().keys()}"
            )
            hw = AcceleratorInfo(
                name=acc_info.get("name"),
                vendor=acc_info.get("vendor"),
                hbm_bandwidth_gb=acc_info.get("hbm_bandwidth_gb"),
                hbm_capacity_gb=acc_info.get("hbm_capacity_gb"),
                inter_node_bandwidth_gb=acc_info.get("inter_node_bandwidth_gb"),
                intra_node_bandwidth_gb=acc_info.get("intra_node_bandwidth_gb"),
                tflops=acc_info.get("tflops"),
            )
        else:
            logger.info(f"Device info initialized: {hw}")
        return hw

    @classmethod
    def get_platform_config(cls) -> PlatformConfig:
        if cls._platform_config is None:
            hw = cls.get_accelerator_info()
            config = cls._get_raw_config()
            platform_config = config.get("platform", {})
            cls._platform_config = PlatformConfig(
                device=hw,
                disk_read_bandwidth_gb=platform_config.get("disk_read_bandwidth_gb"),
                disk_write_bandwidth_gb=platform_config.get("disk_write_bandwidth_gb"),
                memory_read_bandwidth_gb=platform_config.get(
                    "memory_read_bandwidth_gb"
                ),
                memory_write_bandwidth_gb=platform_config.get(
                    "memory_write_bandwidth_gb"
                ),
                num_device_per_node=platform_config.get("num_device_per_node"),
            )

            logger.info(
                f"Platform configuration initialized successfully. {cls._platform_config}"
            )

        return cls._platform_config

    @classmethod
    def set_scheduler_config(cls, config: SchedulerConfig):
        """Apply simulator overrides for data types and the predictor backend."""
        external_config = cls._get_raw_config().get("scheduler", {})
        for field_name in ["backend_name", "backend_version"]:
            field_value = external_config.get(field_name)
            if field_value is not None:
                setattr(config, field_name, field_value)

        for field_name in ["data_type", "kv_cache_data_type"]:
            field_value = external_config.get(field_name)
            if field_value is not None:
                setattr(config, field_name, DataType(field_value))

        cls._scheduler_config = config

    @classmethod
    def get_scheduler_config(cls):
        return cls._scheduler_config

    @classmethod
    def get_ple_admission_model(cls):
        from hisim.time_predictor.ple import PLEAdmissionModel

        ple_config = cls._get_raw_config().get("ple") or {}
        unknown = set(ple_config) - {"admission"}
        if unknown:
            raise ValueError(f"Unknown ple config keys: {sorted(unknown)}")
        return PLEAdmissionModel.from_config(ple_config.get("admission"))

    @classmethod
    def get_inference_time_predictor(
        cls, model: ModelInfo, hw: AcceleratorInfo, sched_config: SchedulerConfig
    ) -> InferTimePredictor:
        config = cls._get_raw_config()
        predictor_config = config.get("predictor", {})
        if predictor_config.get("name") == "linear":
            return LinearTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                base_ms=predictor_config.get("base_ms", 0.0),
                per_token_ms=predictor_config.get("per_token_ms", 0.0),
            )
        elif predictor_config.get("name") == "aiconfigurator":
            from hisim.time_predictor import AIConfiguratorTimePredictor

            database_mode = predictor_config.get("database_mode", "SILICON")
            prefill_scale_factor = predictor_config.get("prefill_scale_factor", 1)
            decode_scale_factor = predictor_config.get("decode_scale_factor", 1)
            prefill_min_latency = predictor_config.get("prefill_min_latency", 0)
            workload_distribution = predictor_config.get(
                "workload_distribution", "balanced"
            )
            enable_oom_check = predictor_config.get("enable_oom_check", False)

            return AIConfiguratorTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=predictor_config.get("database_path"),
                database_mode=database_mode,
                prefill_scale_factor=prefill_scale_factor,
                decode_scale_factor=decode_scale_factor,
                prefill_min_latency=prefill_min_latency,
                workload_distribution=workload_distribution,
                enable_oom_check=enable_oom_check,
            )
        elif predictor_config.get("name") == "ml":
            from hisim.time_predictor.ml import MLTimePredictor

            return MLTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=predictor_config.get("database_path"),
                latency_scale=predictor_config.get("latency_scale", 1.0),
                logprobs_cost_us_per_token=predictor_config.get(
                    "logprobs_cost_us_per_token", 0.0
                ),
                sample_tokens_base_ms=predictor_config.get(
                    "sample_tokens_base_ms", 0.0
                ),
                # Form A (legacy hinge): pass lo/hi/breakpoint only.
                sample_tokens_lo_us_per_token=predictor_config.get(
                    "sample_tokens_lo_us_per_token", 0.0
                ),
                sample_tokens_hi_us_per_token=predictor_config.get(
                    "sample_tokens_hi_us_per_token", 0.0
                ),
                sample_tokens_breakpoint_tokens=predictor_config.get(
                    "sample_tokens_breakpoint_tokens", 0
                ),
                # Form B (mechanistic): pass a/b only. Mutually exclusive
                # with the hinge keys; MLTimePredictor rejects both together.
                sample_tokens_a_ms_per_1k_ext=predictor_config.get(
                    "sample_tokens_a_ms_per_1k_ext", 0.0
                ),
                sample_tokens_b_ms_per_1g_ext_past=predictor_config.get(
                    "sample_tokens_b_ms_per_1g_ext_past", 0.0
                ),
            )
        else:
            raise ValueError(
                f"Unknown predictor name: {predictor_config.get('name')}. "
                "Supported: linear, aiconfigurator, ml"
            )
