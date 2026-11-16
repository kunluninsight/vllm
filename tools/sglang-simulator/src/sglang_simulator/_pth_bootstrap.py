import os

# The same wheel serves either real collection or CPU simulation, never both.
if (os.environ.get("SIM_COLLECTOR_ENABLE", "").lower() in ("1", "true", "yes", "on")
        and os.environ.get("SGLANG_SIMULATOR_ENABLE", "").lower() in ("1", "true", "yes", "on")):
    raise SystemExit("HiSim collector and simulator cannot run in the same interpreter")

import os
from sglang_simulator.hook import install_class_hooks, install_module_hooks

_enabled = os.environ.get("SIM_COLLECTOR_ENABLE", "").lower() in ("1", "true", "yes", "on")
_mode = os.environ.get("SIM_COLLECTOR_MODE", "legacy")

if _enabled and _mode == "pp_v3":
    from sglang_simulator.collector.vllm_hook.trace_collection_hooks import HOOKS
    from sglang_simulator.request_boundaries import HOOKS as request_boundary_hooks
    install_class_hooks([*HOOKS, *request_boundary_hooks])
elif _enabled and _mode == "pp_v2":
    from sglang_simulator.collector.vllm_hook.trace_v2_hooks import HOOKS
    install_class_hooks(HOOKS)
elif _enabled and _mode == "legacy":
    from sglang_simulator.collector.vllm_hook.worker_hook import (
        C_VLLMEngineArgsHook,
        C_WorkerHook,
        C_TpuWorkerHook,
        C_MultiprocExecutorProfileHook,
        C_SchedulerHook,
        # torchtpu subclass overrides Scheduler.schedule; re-apply on it.
        C_TpuSchedulerHook,
        C_EngineCoreHook,
        # TPU pipeline: hand-off wait vs this stage's own forward.
        C_PPWaveHook,
        # TPU pipeline lazy-push step loop: the only observable per-step
        # forward time (EngineCore.step is replaced and non-blocking there).
        C_LazyPushStepperHook,
        C_V6dWorkerHook,
        C_V6dSwapHook,
        # v6d connector ships as top-level kvs_connector/ on this image.
        C_KvsV6dWorkerHook,
        C_KvsV6dSwapHook,
    )

    from sglang_simulator.collector.dashserving import (
        M_DashservingEntrypointHook,
    )

    install_module_hooks(
        [
            M_DashservingEntrypointHook,
        ]
    )

    install_class_hooks(
        [
            C_VLLMEngineArgsHook,
            C_WorkerHook,
            # torchtpu fork: same collection body, different (module, class).
            C_TpuWorkerHook,
            # Multi-node PP followers keep a bare MultiprocExecutor as their
            # backend engine; dashllm calls .start_profile() on it directly.
            C_MultiprocExecutorProfileHook,
            C_SchedulerHook,
            # torchtpu subclass overrides Scheduler.schedule; re-apply on it.
            C_TpuSchedulerHook,
            C_EngineCoreHook,
            # TPU pipeline: hand-off wait vs this stage's own forward.
            C_PPWaveHook,
            # TPU pipeline lazy-push step loop: the only observable per-step
            # forward time (EngineCore.step is replaced and non-blocking there).
            C_LazyPushStepperHook,
            C_V6dWorkerHook,
            C_V6dSwapHook,
            # v6d connector ships as top-level kvs_connector/ on this image.
            C_KvsV6dWorkerHook,
            C_KvsV6dSwapHook,
        ]
    )


import json
import os
import sys

if os.environ.get("SGLANG_SIMULATOR_ENABLE", "").lower() in ("1", "true", "yes", "on"):
    print("[HISIM_BOOTSTRAP] " + json.dumps({
        "pid": os.getpid(), "ppid": os.getppid(), "stage": "pth_entry",
        "executable": sys.executable, "bootstrap_origin": __file__,
        "isolated": sys.flags.isolated, "no_site": sys.flags.no_site,
        "ignore_environment": sys.flags.ignore_environment,
        "vllm_present": "vllm" in sys.modules,
    }), file=sys.stderr, flush=True)
    from sglang_simulator.simulation.vllm.startup import init_hook as _vllm_init_hook
    from sglang_simulator.simulation.sglang.startup import init_hook as _sglang_init_hook

    from sglang_simulator.simulation.vllm.v6d.v6d_capacity import install_v6d_runtime_hooks

    _vllm_init_hook()
    _sglang_init_hook()
    install_v6d_runtime_hooks()
