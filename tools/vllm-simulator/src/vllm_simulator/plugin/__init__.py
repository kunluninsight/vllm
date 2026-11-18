# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plugin discovery must stay independent of vLLM's import lifecycle."""

import os


def register_platform() -> str | None:
    if "simulator" in os.getenv("VLLM_PLUGINS", "").split(","):
        return "vllm_simulator.plugin.platform.SimulatorPlatform"
    return None
