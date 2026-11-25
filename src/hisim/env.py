"""Central environment configuration, safe to import during Python startup.

Reads remain lazy: callers decide when to snapshot settings. Importing this
module never imports inference frameworks or changes the process environment.
"""

import json
import logging
import os


class Envs:
    ENABLE = "HISIM_ENABLE"
    CONFIG_PATH = "HISIM_CONFIG_PATH"
    CONFIG_JSON = "HISIM_CONFIG_JSON"
    OUTPUT_DIR = "HISIM_OUTPUT_DIR"
    OUTPUT_MODE = "HISIM_OUTPUT_MODE"
    RECORD_STATS = "HISIM_RECORD_STATS"
    LOG_LEVEL = "HISIM_LOG_LEVEL"
    MAX_TOKENS = "HISIM_MAX_TOKENS"
    BW_PROFILE = "HISIM_BW_PROFILE"
    STORE_POLL_MS = "HISIM_STORE_POLL_MS"
    STORE_RANK_SYNC_MS = "HISIM_STORE_RANK_SYNC_MS"
    SEG1_FLOOR_MS = "HISIM_SEG1_FLOOR_MS"
    SEG1_PER_BLK_MS = "HISIM_SEG1_PER_BLK_MS"
    DP_MAX_REQUESTS = "HISIM_DP_MAX_REQUESTS"

    # Tair KVCM / MemPool / V6D P2P / KVT transfer modeling controls.
    TAIR_KVCM_READ_BANDWIDTH_GB = "HISIM_TAIR_KVCM_READ_BANDWIDTH_GB"
    TAIR_KVCM_READ_BASE_S = "HISIM_TAIR_KVCM_READ_BASE_S"
    TAIR_KVCM_SSD_READ_BANDWIDTH_GB = "HISIM_TAIR_KVCM_SSD_READ_BANDWIDTH_GB"
    TAIR_KVCM_SSD_READ_BASE_S = "HISIM_TAIR_KVCM_SSD_READ_BASE_S"
    TAIR_KVCM_WRITE_BANDWIDTH_GB = "HISIM_TAIR_KVCM_WRITE_BANDWIDTH_GB"
    TAIR_KVCM_WRITE_BASE_S = "HISIM_TAIR_KVCM_WRITE_BASE_S"
    TAIR_MEMPOOL_META_URL = "HISIM_TAIR_MEMPOOL_META_URL"
    V6D_P2P_BANDWIDTH_GB = "HISIM_V6D_P2P_BANDWIDTH_GB"
    V6D_P2P_BASE_S = "HISIM_V6D_P2P_BASE_S"
    V6D_LOAD_LANES = "HISIM_V6D_LOAD_LANES"
    V6D_STORE_LANES = "HISIM_V6D_STORE_LANES"
    V6D_LOAD_BASE_S = "HISIM_V6D_LOAD_BASE_S"
    V6D_LOAD_BANDWIDTH_GB = "HISIM_V6D_LOAD_BANDWIDTH_GB"
    V6D_QUERY_MIN_MS = "HISIM_V6D_QUERY_MIN_MS"
    KVT_HANDOFF_BASE_S = "HISIM_KVT_HANDOFF_BASE_S"
    KVT_HANDOFF_BANDWIDTH_GB = "HISIM_KVT_HANDOFF_BANDWIDTH_GB"
    KVT_HANDOFF_LANES = "HISIM_KVT_HANDOFF_LANES"
    KVT_HANDOFF_BYTES_PER_TOKEN = "HISIM_KVT_HANDOFF_BYTES_PER_TOKEN"

    # Existing integration-owned names retain their upstream spelling.
    V6D_HITSOURCE_LOG = "V6D_HITSOURCE_LOG"
    V6D_KEYSOURCE_LOG = "V6D_KEYSOURCE_LOG"

    _TRUTHY = ("1", "true", "yes", "on")
    _DEFAULT_OUTPUT_DIR = "/tmp/hisim/output/"

    @classmethod
    def enabled(cls) -> bool:
        return os.getenv(cls.ENABLE, "").lower() in cls._TRUTHY

    @classmethod
    def config_path(cls, required: bool = True) -> str | None:
        path = os.getenv(cls.CONFIG_PATH)
        if not path or not os.path.exists(path):
            if required:
                raise RuntimeError(
                    f"The configuration path is not set or does not exist({path}). "
                    f"Please set it using the system variable {cls.CONFIG_PATH}"
                )
            return None
        return path

    @classmethod
    def config_json(cls) -> str | None:
        return os.getenv(cls.CONFIG_JSON, "").strip() or None

    @classmethod
    def load_config(cls) -> dict:
        """Load the simulator config; HISIM_CONFIG_JSON overrides HISIM_CONFIG_PATH."""
        text = cls.config_json()
        if text is not None:
            try:
                config = json.loads(text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in {cls.CONFIG_JSON}: {e}") from e
        else:
            path = cls.config_path(required=False)
            if path is None:
                raise RuntimeError(
                    f"The simulator configuration is not set. Set {cls.CONFIG_JSON} "
                    f"to a JSON string or {cls.CONFIG_PATH} to an existing file "
                    f"({os.getenv(cls.CONFIG_PATH)})."
                )
            with open(path) as f:
                config = json.load(f)
        if not isinstance(config, dict):
            raise TypeError("The simulator configuration must be a JSON object.")
        return config

    @classmethod
    def set_config_path(cls, path: str) -> None:
        os.environ[cls.CONFIG_PATH] = os.fspath(path)

    @classmethod
    def output_dir(cls) -> str:
        path = os.path.realpath(os.getenv(cls.OUTPUT_DIR, cls._DEFAULT_OUTPUT_DIR))
        if os.path.exists(path) and os.path.isfile(path):
            logging.getLogger("hisim").error(
                "The metrics output path, %s, exists and is a file.", path
            )
            raise RuntimeError(f"{path} exists but is not a directory.")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def worker_output_dir(cls, name: str) -> str:
        """Set the inherited worker default without creating the directory."""
        path = os.environ.setdefault(cls.OUTPUT_DIR, f"/tmp/hisim/{name}")
        return os.path.realpath(path)

    @classmethod
    def simulation_mode(cls) -> str:
        mode = os.getenv(cls.OUTPUT_MODE, "OFFLINE").upper()
        assert mode in ("BLOCKING", "OFFLINE")
        return mode

    @classmethod
    def server_output_mode(cls) -> str:
        """Preserve the launcher's display default without changing the mode."""
        return os.getenv(cls.OUTPUT_MODE, "BLOCKING")

    @classmethod
    def set_output_mode(cls, mode: str) -> None:
        os.environ[cls.OUTPUT_MODE] = mode

    @classmethod
    def record_stats(cls) -> bool:
        return os.getenv(cls.RECORD_STATS, "0").lower() in cls._TRUTHY

    @classmethod
    def log_level(cls) -> int:
        name = os.getenv(cls.LOG_LEVEL, "INFO").upper()
        return getattr(logging, name, logging.INFO)

    @classmethod
    def max_tokens(cls) -> int | None:
        value = os.getenv(cls.MAX_TOKENS)
        return int(value) if value is not None else None

    @classmethod
    def dp_max_requests(cls) -> int | None:
        """Per-rank request capacity of the DP payload; None uses max_num_seqs."""
        value = os.getenv(cls.DP_MAX_REQUESTS)
        return int(value) if value is not None else None

    @classmethod
    def bandwidth_profile_path(cls) -> str:
        return os.getenv(cls.BW_PROFILE, "").strip()

    @classmethod
    def store_poll_ms(cls, default: float) -> float:
        value = os.getenv(cls.STORE_POLL_MS)
        return float(value) if value is not None else default

    @classmethod
    def store_rank_sync_ms(cls, default: float) -> float:
        value = os.getenv(cls.STORE_RANK_SYNC_MS)
        return float(value) if value is not None else default

    @classmethod
    def seg1_latency_parameters(
        cls, default_floor_ms: float, default_per_block_ms: float
    ) -> tuple[float, float]:
        floor = os.getenv(cls.SEG1_FLOOR_MS)
        per_block = os.getenv(cls.SEG1_PER_BLK_MS)
        # An override of either setting replaces the whole profile pair.
        if floor is not None or per_block is not None:
            return float(floor or 0), float(per_block or 0)
        return default_floor_ms, default_per_block_ms

    @classmethod
    def v6d_hitsource_enabled(cls) -> bool:
        return os.getenv(cls.V6D_HITSOURCE_LOG, "1") not in ("0", "false", "False")

    @classmethod
    def v6d_keysource_enabled(cls) -> bool:
        return os.getenv(cls.V6D_KEYSOURCE_LOG, "1") not in ("0", "false", "False")

    @classmethod
    def vllm_async_registration_enabled(cls) -> bool:
        """Consult vLLM's native parser only when the integration requests it."""
        from vllm import envs

        return envs.VLLM_V6D_ASYNC_REGISTER

    @staticmethod
    def _non_negative_float(name: str, default: float = 0.0) -> float:
        import math

        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            val = float(raw)
        except ValueError:
            raise ValueError(
                f"{name} must be a non-negative finite number"
            ) from None
        if not math.isfinite(val) or val < 0:
            raise ValueError(f"{name} must be a non-negative finite number")
        return val

    @staticmethod
    def _non_negative_int(name: str, default: int = 0) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be a non-negative integer") from None
        if val < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return val

    @classmethod
    def tair_kvcm_read_bandwidth_bps(cls) -> float:
        return cls._non_negative_float(cls.TAIR_KVCM_READ_BANDWIDTH_GB) * 1e9

    @classmethod
    def tair_kvcm_read_base_s(cls) -> float:
        return cls._non_negative_float(cls.TAIR_KVCM_READ_BASE_S)

    @classmethod
    def tair_kvcm_ssd_read_bandwidth_bps(cls) -> float:
        return cls._non_negative_float(cls.TAIR_KVCM_SSD_READ_BANDWIDTH_GB) * 1e9

    @classmethod
    def tair_kvcm_ssd_read_base_s(cls) -> float:
        return cls._non_negative_float(cls.TAIR_KVCM_SSD_READ_BASE_S)

    @classmethod
    def tair_kvcm_write_bandwidth_bps(cls) -> float:
        return cls._non_negative_float(cls.TAIR_KVCM_WRITE_BANDWIDTH_GB) * 1e9

    @classmethod
    def tair_kvcm_write_base_s(cls) -> float:
        return cls._non_negative_float(cls.TAIR_KVCM_WRITE_BASE_S)

    @classmethod
    def tair_mempool_meta_url(cls) -> str:
        return os.getenv(cls.TAIR_MEMPOOL_META_URL, "").strip().rstrip("/")

    @classmethod
    def v6d_p2p_bandwidth_bps(cls) -> float:
        return cls._non_negative_float(cls.V6D_P2P_BANDWIDTH_GB) * 1e9

    @classmethod
    def v6d_p2p_base_s(cls) -> float:
        return cls._non_negative_float(cls.V6D_P2P_BASE_S)

    @classmethod
    def v6d_load_lanes(cls) -> int:
        return cls._non_negative_int(cls.V6D_LOAD_LANES, 0)

    @classmethod
    def v6d_store_lanes(cls) -> int:
        return cls._non_negative_int(cls.V6D_STORE_LANES, 0)

    @classmethod
    def v6d_load_base_s(cls) -> float:
        return cls._non_negative_float(cls.V6D_LOAD_BASE_S)

    @classmethod
    def v6d_load_bandwidth_bps(cls) -> float:
        return cls._non_negative_float(cls.V6D_LOAD_BANDWIDTH_GB) * 1e9

    @classmethod
    def v6d_query_min_s(cls) -> float:
        return cls._non_negative_float(cls.V6D_QUERY_MIN_MS) / 1000.0

    @classmethod
    def kvt_handoff_base_s(cls) -> float:
        return cls._non_negative_float(cls.KVT_HANDOFF_BASE_S)

    @classmethod
    def kvt_handoff_bandwidth_bps(cls) -> float:
        return cls._non_negative_float(cls.KVT_HANDOFF_BANDWIDTH_GB) * 1e9

    @classmethod
    def kvt_handoff_lanes(cls) -> int:
        lanes = cls._non_negative_int(cls.KVT_HANDOFF_LANES, 8)
        return max(1, lanes)

    @classmethod
    def kvt_handoff_bytes_per_token(cls) -> int:
        return cls._non_negative_int(cls.KVT_HANDOFF_BYTES_PER_TOKEN, 0)
