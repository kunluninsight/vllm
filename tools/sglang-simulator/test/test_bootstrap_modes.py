"""Keep collector and simulator activation independent in the integrated wheel."""
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

BOOTSTRAP = Path(__file__).parents[1] / "src/sglang_simulator/_pth_bootstrap.py"


@pytest.mark.parametrize("collector,simulator,expected", [
    ("0", "0", []), ("1", "0", ["collector", "boundaries"]),
    ("0", "1", ["vllm", "sglang", "v6d"]),
])
def test_activation(monkeypatch, collector, simulator, expected):
    calls = []
    monkeypatch.setenv("SIM_COLLECTOR_ENABLE", collector)
    monkeypatch.setenv("SIM_COLLECTOR_MODE", "pp_v3")
    monkeypatch.setenv("SGLANG_SIMULATOR_ENABLE", simulator)
    modules = {
        "sglang_simulator.hook": dict(install_class_hooks=calls.extend, install_module_hooks=calls.extend),
        "sglang_simulator.collector.vllm_hook.trace_collection_hooks": dict(HOOKS=["collector"]),
        "sglang_simulator.request_boundaries": dict(HOOKS=["boundaries"]),
        "sglang_simulator.simulation.vllm.startup": dict(init_hook=lambda: calls.append("vllm")),
        "sglang_simulator.simulation.sglang.startup": dict(init_hook=lambda: calls.append("sglang")),
        "sglang_simulator.simulation.vllm.v6d.v6d_capacity": dict(install_v6d_runtime_hooks=lambda: calls.append("v6d")),
    }
    for name, fields in modules.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    runpy.run_path(str(BOOTSTRAP))
    assert calls == expected


def test_conflicting_modes_stop_before_hook_installation(monkeypatch):
    monkeypatch.setenv("SIM_COLLECTOR_ENABLE", "true")
    monkeypatch.setenv("SGLANG_SIMULATOR_ENABLE", "1")
    with pytest.raises(SystemExit, match="cannot run in the same interpreter"):
        runpy.run_path(str(BOOTSTRAP))


def test_sim_frontend_recorder_with_collector_also_installed(tmp_path, monkeypatch):
    import os
    import subprocess

    env = dict(os.environ, SIM_COLLECTOR_ENABLE="0", SGLANG_SIMULATOR_ENABLE="0",
               HISIM_LIFECYCLE_ENABLE="1", SGLANG_SIMULATOR_LIFECYCLE_DIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", """
from pathlib import Path
from sglang_simulator import request_boundaries as b
assert (Path(b.__file__).parent / 'collector').is_dir()
b._emit('test.frontend', marker=1)
assert b._SIM_RECORDER is not None and b._SIM_RECORDER.flush()
assert b._TRACE is None
b._close_sim()
"""], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
