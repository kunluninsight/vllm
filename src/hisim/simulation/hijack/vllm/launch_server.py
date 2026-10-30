"""Launch vLLM's OpenAI server with HiSim hooks.

Pass vLLM arguments plus --sim-config-path, or set HISIM_CONFIG_PATH or
HISIM_CONFIG_JSON.
Set HISIM_OUTPUT_MODE=BLOCKING for wall-clock serving; the simulation default
is OFFLINE even though the launcher's status message defaults to BLOCKING.
"""

import os

from hisim.env import Envs

# Configure framework defaults before importing vLLM.
os.environ.setdefault("HISIM_ENABLE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

from hisim.simulation.hijack.startup import init_hook
from hisim.utils import get_logger

init_hook()

logger = get_logger()


if __name__ == "__main__":
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM Simulation Server (OpenAI-compatible)"
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--sim-config-path",
        type=str,
        default=None,
        help="Path to simulation JSON config (same as HISIM_CONFIG_PATH).",
    )

    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # An existing environment config takes precedence over the CLI path.
    config_path = Envs.config_path(required=False)
    if Envs.config_json():
        logger.info(f"Using config from env: {Envs.CONFIG_JSON}")
    elif config_path:
        logger.info(f"Using config from env: {config_path}")
    elif args.sim_config_path:
        Envs.set_config_path(args.sim_config_path)
        logger.info(f"Using config from arg: {args.sim_config_path}")

    logger.info(
        "Starting vLLM simulation server in %s mode.",
        Envs.server_output_mode(),
    )

    uvloop.run(run_server(args))
