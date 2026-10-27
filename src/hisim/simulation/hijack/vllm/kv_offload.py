"""Complete native CPU-offload jobs without allocating CUDA transfer workers.

Support SimpleCPUOffloadWorker and OffloadingConnectorWorker. Native scheduler
cache accounting receives completed load/store IDs; no payload bytes move.
"""

from hisim.hook import BaseHook
from hisim.utils import get_logger

logger = get_logger()


class C_VLLMSimpleCPUOffloadWorkerHook(BaseHook):
    """Hook SimpleCPUOffloadWorker to skip CUDA and auto-complete transfers."""

    HOOK_CLASS_NAME = "SimpleCPUOffloadWorker"
    HOOK_MODULE_NAME = "vllm.v1.simple_kv_offload.worker"

    @classmethod
    def hook(cls, target):
        def override_init(self, vllm_config, kv_cache_config, cpu_capacity_bytes):
            """Skip pinned memory allocation and CUDA stream creation."""
            self._captured_metadata = None
            self._completed_store_events: dict[int, int] = {}
            logger.info(
                "[vLLM Hijack] SimpleCPUOffloadWorker.__init__: "
                "skipped CUDA allocation (simulation mode)"
            )

        def override_register_kv_caches(self, kv_caches):
            """Skip tensor registration because no payload transfers are issued."""
            pass

        def override_bind_connector_metadata(self, metadata):
            """Capture metadata so we can report events as completed."""
            self._captured_metadata = metadata

        def override_clear_connector_metadata(self):
            """Clear captured metadata."""
            self._captured_metadata = None

        def override_handle_preemptions(self, kv_connector_metadata):
            """No-op: no in-flight transfers to flush."""
            pass

        def override_get_finished(self, finished_req_ids):
            """Report loads as received; publish store IDs through worker metadata."""
            meta = self._captured_metadata
            finished_recving = set()

            if meta is not None and meta.load_event >= 0:
                for reqs in meta.load_event_to_reqs.values():
                    finished_recving.update(reqs)

            if meta is not None and meta.store_event >= 0:
                self._completed_store_events[meta.store_event] = 1

            return None, finished_recving or None

        def override_build_connector_worker_meta(self):
            """Report all pending store events as completed."""
            from vllm.v1.simple_kv_offload.metadata import (
                SimpleCPUOffloadWorkerMetadata,
            )

            if not self._completed_store_events:
                return None
            meta = SimpleCPUOffloadWorkerMetadata(
                completed_store_events=self._completed_store_events,
            )
            self._completed_store_events = {}
            return meta

        target.__init__ = override_init
        target.register_kv_caches = override_register_kv_caches
        target.bind_connector_metadata = override_bind_connector_metadata
        target.clear_connector_metadata = override_clear_connector_metadata
        target.handle_preemptions = override_handle_preemptions
        target.get_finished = override_get_finished
        target.build_connector_worker_meta = override_build_connector_worker_meta


class C_VLLMOffloadingConnectorHook(BaseHook):
    """Resolve the native CPU capacity from finalized KV tensor accounting."""

    HOOK_CLASS_NAME = "OffloadingConnector"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector"
    )

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def initialize(self, vllm_config, role, kv_cache_config=None):
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config
            budget = extra.get("kv_bytes_per_rank")
            if (
                not extra.get("num_cpu_blocks")
                and budget is not None
                and kv_cache_config is not None
            ):
                gpu_blocks = kv_cache_config.num_blocks
                tensor_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
                gpu_block_size = vllm_config.cache_config.block_size
                cpu_block_size = int(extra.get("block_size", gpu_block_size))
                if gpu_blocks <= 0 or tensor_bytes <= 0:
                    raise ValueError(
                        "CPU offloading requires nonempty KV cache accounting"
                    )
                if (
                    gpu_block_size <= 0
                    or cpu_block_size <= 0
                    or cpu_block_size % gpu_block_size
                ):
                    raise ValueError(
                        "CPU offload block size must be a positive multiple of GPU block size"
                    )
                bytes_per_block = (tensor_bytes // gpu_blocks) * (
                    cpu_block_size // gpu_block_size
                )
                if bytes_per_block <= 0:
                    raise ValueError(
                        "CPU offloading requires positive bytes per KV block"
                    )
                num_cpu_blocks = int(budget // bytes_per_block)
                if num_cpu_blocks <= 0:
                    raise ValueError("CPU offload budget cannot hold one KV block")
                extra["num_cpu_blocks"] = num_cpu_blocks
                logger.info(
                    "[vLLM Hijack] CPU offload capacity: %d blocks, %d bytes/block",
                    num_cpu_blocks,
                    bytes_per_block,
                )
            return original_init(self, vllm_config, role, kv_cache_config)

        target.__init__ = initialize


class C_VLLMOffloadingConnectorWorkerHook(BaseHook):
    """Hook native request-based and job-based offload workers without CUDA."""

    HOOK_CLASS_NAME = "OffloadingConnectorWorker"
    HOOK_MODULE_NAME = r"vllm\.distributed\.kv_transfer\.kv_connector\.v1\.(offloading\.worker|offloading_connector)$"
    REGEX = True

    @classmethod
    def _hook_request_api(cls, target):
        def initialize(self, spec):
            self.spec = spec
            self.worker = None
            self._sim_loads = set()
            self._sim_stores = set()

        def register(self, *args, **kwargs):
            return None

        def start_load(self, metadata):
            self._sim_loads.update(metadata.reqs_to_load)

        def start_store(self, metadata):
            self._sim_stores.update(metadata.reqs_to_store)

        def get_finished(self, finished_req_ids):
            # Release native store protection only after generation finishes.
            sending = self._sim_stores.intersection(finished_req_ids or ())
            self._sim_stores.difference_update(sending)
            receiving, self._sim_loads = self._sim_loads, set()
            return sending, receiving

        target.__init__ = initialize
        target.register_kv_caches = register
        target.register_cross_layers_kv_cache = register
        target.start_load_kv = start_load
        target.start_store_kv = start_store
        target.get_finished = get_finished

    @classmethod
    def hook(cls, target):
        if hasattr(target, "start_load_kv"):
            cls._hook_request_api(target)
            return

        def override_init(self, spec):
            """Skip OffloadingWorker (CUDA streams) creation."""
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
                OffloadingWorkerMetadata,
            )
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
                OffloadingConnectorStats,
            )

            self.spec = spec
            self.worker = None  # no real worker
            self.kv_connector_stats = OffloadingConnectorStats()
            self._load_jobs: dict[int, str] = {}
            self._unsubmitted_store_jobs: list = []
            self._connector_worker_meta = OffloadingWorkerMetadata()
            logger.info(
                "[vLLM Hijack] OffloadingConnectorWorker.__init__: "
                "skipped CUDA allocation (simulation mode)"
            )

        def override_register_kv_caches(self, kv_caches):
            """Skip tensor registration because no payload transfers are issued."""
            pass

        def override_register_cross_layers_kv_cache(self, kv_cache, attn_backend):
            """No-op: no cross-layer tensors needed."""
            pass

        def override_handle_preemptions(self, kv_connector_metadata):
            """Discard unsubmitted jobs; no payload transfers need flushing."""
            self._unsubmitted_store_jobs.clear()

        def override_start_kv_transfers(self, metadata):
            """Immediately mark all load jobs as complete."""
            self._unsubmitted_store_jobs.clear()
            for job_id, entry in metadata.load_jobs.items():
                self._load_jobs[job_id] = entry.req_id

        def override_prepare_store_kv(self, metadata):
            """Track store jobs for immediate completion reporting."""
            for job_id in metadata.store_jobs:
                self._connector_worker_meta.mark_completed(job_id)

        def override_get_finished(self, finished_req_ids):
            """Report all loads as immediately finished."""
            finished_recving: set[str] = set()
            for job_id, req_id in self._load_jobs.items():
                self._connector_worker_meta.mark_completed(job_id)
                finished_recving.add(req_id)
            self._load_jobs.clear()
            return set(), finished_recving

        def override_build_connector_worker_meta(self):
            """Return completed job IDs since last call."""
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
                OffloadingWorkerMetadata,
            )

            if not self._connector_worker_meta.completed_jobs:
                return None
            meta = self._connector_worker_meta
            self._connector_worker_meta = OffloadingWorkerMetadata()
            return meta

        def override_shutdown(self):
            """Clear tracked jobs; no transfer worker needs shutdown."""
            self._unsubmitted_store_jobs.clear()
            self._load_jobs.clear()

        target.__init__ = override_init
        target.register_kv_caches = override_register_kv_caches
        target.register_cross_layers_kv_cache = override_register_cross_layers_kv_cache
        target.handle_preemptions = override_handle_preemptions
        target.start_kv_transfers = override_start_kv_transfers
        target.prepare_store_kv = override_prepare_store_kv
        target.get_finished = override_get_finished
        target.build_connector_worker_meta = override_build_connector_worker_meta
        target.shutdown = override_shutdown
