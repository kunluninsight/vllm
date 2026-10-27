"""Loaded by hisim.pth at every Python interpreter startup.

Installs the simulation hooks only when HISIM_ENABLE is truthy
(1/true/yes/on); otherwise this is a no-op.
"""

from hisim.env import Envs

if Envs.enabled():
    from hisim.simulation.hijack.startup import init_hook

    init_hook()
