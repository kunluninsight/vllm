import os

from vllm_simulator.utils.logger import get_logger

logger = get_logger()


class Envs:
    @classmethod
    def config_path(cls) -> str:
        VLLM_SIMULATOR_CONFIG_PATH = os.getenv("VLLM_SIMULATOR_CONFIG_PATH")
        if not VLLM_SIMULATOR_CONFIG_PATH or not os.path.exists(
            VLLM_SIMULATOR_CONFIG_PATH
        ):
            raise RuntimeError(
                f"The configuration path is not set or does not exist({VLLM_SIMULATOR_CONFIG_PATH}). Please set it using the system variable VLLM_SIMULATOR_CONFIG_PATH"
            )
        return VLLM_SIMULATOR_CONFIG_PATH

    @classmethod
    def output_dir(cls) -> str:
        VLLM_SIMULATOR_OUTPUT_DIR = os.getenv(
            "VLLM_SIMULATOR_OUTPUT_DIR", "/tmp/vllm_simulator/output/"
        )
        VLLM_SIMULATOR_OUTPUT_DIR = os.path.realpath(VLLM_SIMULATOR_OUTPUT_DIR)
        if os.path.exists(VLLM_SIMULATOR_OUTPUT_DIR) and os.path.isfile(
            VLLM_SIMULATOR_OUTPUT_DIR
        ):
            logger.error(
                f"The metrics output path, {VLLM_SIMULATOR_OUTPUT_DIR}, exists and is a file."
            )
            raise RuntimeError(
                f"{VLLM_SIMULATOR_OUTPUT_DIR} exists but is not a directory."
            )
        os.makedirs(VLLM_SIMULATOR_OUTPUT_DIR, exist_ok=True)
        return VLLM_SIMULATOR_OUTPUT_DIR

    @classmethod
    def simulation_mode(cls) -> str:
        VLLM_SIMULATOR_OUTPUT_MODE = os.getenv(
            "VLLM_SIMULATOR_OUTPUT_MODE", "OFFLINE"
        ).upper()
        assert VLLM_SIMULATOR_OUTPUT_MODE in ("BLOCKING", "OFFLINE")
        return VLLM_SIMULATOR_OUTPUT_MODE

    @classmethod
    def num_warmup(cls) -> int:
        # The number of warmup requests.
        return int(os.getenv("VLLM_SIMULATOR_NUM_WARMUP", "0"))
