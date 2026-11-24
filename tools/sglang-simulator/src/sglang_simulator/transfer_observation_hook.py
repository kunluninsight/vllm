"""Collector-only worker observer; simulator composes this at its existing hook."""
from functools import wraps
from sglang_simulator.hook import BaseHook
from sglang_simulator.kv_observation import install_worker, worker_emitter


def observe(callback):
    try:
        return callback()
    except BaseException:
        return None


class C_TransferWorkerObservationHook(BaseHook):
    HOOK_MODULE_NAME = r"^(kvs_connector\.v6d_object_connector|vllm_torchtpu\.distributed\.kv_transfer\.tpu_connector)$"
    HOOK_CLASS_NAME = r"^(V6dObjectConnectorWorker|TPURaidenConnectorWorker)$"
    REGEX = True

    @classmethod
    def hook(cls, target):
        native = target.__init__
        if getattr(native, "_hisim_transfer_constructor", False):
            return

        @wraps(native)
        def initialize(self, *args, **kwargs):
            value = native(self, *args, **kwargs)
            def install():
                emit = worker_emitter()
                if emit is not None:
                    install_worker(self, emit, observe)
            observe(install)
            return value
        initialize._hisim_transfer_constructor = True
        target.__init__ = initialize
