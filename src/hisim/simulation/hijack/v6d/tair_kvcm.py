"""Keep KVCM control-plane transfers while bypassing native payload I/O."""

import logging

from hisim.hook import BaseHook

logger = logging.getLogger("hisim")


class _NoopTransferClient:
    """Stands in for kvcm_py_client.TransferClient, whose only uses are stubbed below."""

    def close(self):
        pass


class C_TairKVCMTransferRuntimeHook(BaseHook):
    HOOK_CLASS_NAME = "TairKVCMTransferRuntime"
    HOOK_MODULE_NAME = "v6d.common.tair_kvcm.transfer"

    @classmethod
    def hook(cls, target):
        # Skip the native client: it needs a local tair_mempool_server (TENT on CPU
        # hosts), otherwise init fails with "kvcm_py_client.TransferClient.Create failed"
        # and v6d silently stops reporting BLOCK_ADD / writing to KVCM.
        def create_transfer_client(self):
            return _NoopTransferClient()

        def batch_load(self, uris, pointers, sizes):
            if not (len(uris) == len(pointers) == len(sizes)):
                raise ValueError("uris, pointers, and sizes length mismatch")
            if not uris:
                return True
            from v6d.common.tair_kvcm.transfer import _record_transfer_result

            _record_transfer_result("read", True)
            return True

        def batch_save(self, uris, pointers, sizes):
            if not (len(uris) == len(pointers) == len(sizes)):
                raise ValueError("uris, pointers, and sizes length mismatch")
            if not uris:
                return []
            from v6d.common.tair_kvcm.transfer import _record_transfer_result

            _record_transfer_result("write", True)
            return [True] * len(uris)

        target._create_transfer_client = create_transfer_client
        target.batch_load_to_memory = batch_load
        target.batch_save_from_memory = batch_save
        logger.info(
            "[v6d-sim] C_TairKVCMTransferRuntimeHook installed: "
            "native KVCM transfer client and payload I/O stubbed; control-plane transfers preserved"
        )
