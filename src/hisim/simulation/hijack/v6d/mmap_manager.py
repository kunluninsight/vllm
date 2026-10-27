"""Native v6d client mmap hooks for metadata-only simulation."""

from __future__ import annotations

import os

from hisim.hook import BaseHook
from hisim.utils import get_logger

logger = get_logger()


class C_V6dMmapManagerHook(BaseHook):
    """Keep the native connection socket without mapping or registering payloads.

    base_addr=0 is accounting-only: callers must never dereference blob data.
    The socket and declared map size preserve connection bookkeeping.
    """

    HOOK_CLASS_NAME = "ClientV6dMmapManager"
    HOOK_MODULE_NAME = "v6d.client.peers.vineyard.mmap_manager"

    @classmethod
    def hook(cls, target):
        from v6d.client.peers.vineyard.mmap_manager import MmapInfo

        def override_create_mmap(self, socket_path: str, is_lazy_strategy: bool):
            from v6d.common.transfer import _vineyard_connect

            fd, map_size, offset, socket = _vineyard_connect(socket_path)
            os.close(fd)  # Keep the socket alive, but no mapping owns this descriptor.
            logger.info(
                "[V6D Hijack] sim mmap: skip bulkstore mmap for %s "
                "(map_size=%d MB, no client data-plane dereference)",
                socket_path,
                map_size // (1 << 20),
            )
            return MmapInfo(
                socket_path=socket_path,
                socket=socket,
                fd=-1,
                base_addr=0,
                map_size=map_size,
                refcount=1,
            )

        target._create_mmap = override_create_mmap
        logger.info(
            "[V6D Hijack] ClientV6dMmapManager hook installed "
            "(no bulkstore mmap population)"
        )
