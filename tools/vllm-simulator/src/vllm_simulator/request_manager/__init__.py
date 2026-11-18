# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-engine request lifecycle management and pluggable admission policies."""

from importlib import import_module

from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.request_manager.dispatcher import (
    BlockingDispatcher,
    ConcurrencyDispatcher,
    ConversationDispatcher,
    RequestDispatcher,
    TimeDispatcher,
)


def create_dispatcher(config) -> RequestDispatcher:
    simulator = SimulatorConfig.from_config(config)
    options = simulator.request_dispatcher.copy()
    blocking = simulator.mode == "BLOCKING"
    name = options.pop("name", "blocking" if blocking else "time")
    factories = {
        "blocking": BlockingDispatcher,
        "time": TimeDispatcher,
        "concurrency": ConcurrencyDispatcher,
        "conversation": ConversationDispatcher,
    }
    if name in factories:
        dispatcher = factories[name](**options)
    elif isinstance(name, str) and ":" in name:
        module, attribute = name.split(":", 1)
        dispatcher = getattr(import_module(module), attribute)(config, **options)
    else:
        raise ValueError(f"Unknown request dispatcher: {name!r}")
    if not isinstance(dispatcher, RequestDispatcher):
        raise TypeError("Dispatcher factories must return RequestDispatcher instances")
    if isinstance(dispatcher, BlockingDispatcher) != blocking:
        raise ValueError(
            "BLOCKING mode requires a BlockingDispatcher; "
            "replay policies require NON_BLOCKING"
        )
    return dispatcher
