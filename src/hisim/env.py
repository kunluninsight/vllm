"""Central environment configuration, safe to import during Python startup.

Reads remain lazy: callers decide when to snapshot settings. Importing this
module never imports inference frameworks or changes the process environment.
"""

import logging
import os


class Envs:
    ENABLE = "HISIM_ENABLE"
    CONFIG_PATH = "HISIM_CONFIG_PATH"
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

    # Existing integration-owned names retain their upstream spelling.
    V6D_HITSOURCE_LOG = "V6D_HITSOURCE_LOG"
    V6D_KEYSOURCE_LOG = "V6D_KEYSOURCE_LOG"
    V6D_SIM_STUB_META_PROBE = "V6D_SIM_STUB_META_PROBE"

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
    def v6d_stub_meta_probe_enabled(cls) -> bool:
        return os.getenv(cls.V6D_SIM_STUB_META_PROBE, "1") not in (
            "0",
            "false",
            "False",
        )

    @classmethod
    def vllm_async_registration_enabled(cls) -> bool:
        """Consult vLLM's native parser only when the integration requests it."""
        from vllm import envs

        return envs.VLLM_V6D_ASYNC_REGISTER
