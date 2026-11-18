# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate native offload event/job protocols into common block transfers."""

from abc import ABC, abstractmethod

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.metadata import (
    INVALID_JOB_ID,
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.sizing import gpu_total_bytes
from vllm_simulator.plugin.kv_transfer.backend import (
    BlockSelection,
    TransferBackend,
    TransferJob,
)
from vllm_simulator.plugin.kv_transfer.capacity import cpu_offloading_spec


class ConnectorAdapter(ABC):
    def __init__(self, kv_cache_config):
        self.backend = TransferBackend()
        self.bytes_per_block = (
            gpu_total_bytes(kv_cache_config) // kv_cache_config.num_blocks
        )
        self.backend.register_region(
            "gpu", kv_cache_config.num_blocks, self.bytes_per_block
        )

    def transfer(self, direction, job_id, gpu_blocks, cpu_blocks) -> TransferJob:
        gpu = BlockSelection("gpu", tuple(int(b) for b in gpu_blocks))
        cpu = BlockSelection("cpu", tuple(int(b) for b in cpu_blocks))
        source, destination = (cpu, gpu) if direction == "load" else (gpu, cpu)
        return TransferJob(
            direction,
            job_id,
            source,
            destination,
            len(gpu.block_ids) * self.bytes_per_block,
        )

    @abstractmethod
    def begin_step(self, metadata) -> None:
        """Submit loads and flush prior work before the model executes."""

    @abstractmethod
    def end_step(self, metadata) -> KVConnectorOutput:
        """Submit or defer stores and translate drained completions."""

    def shutdown(self) -> None:
        self.backend.shutdown()


class SimpleCPUOffloadAdapter(ConnectorAdapter):
    def __init__(self, config, kv_cache_config):
        super().__init__(kv_cache_config)
        extra = config.kv_transfer_config.kv_connector_extra_config
        local_capacity = int(extra["cpu_bytes_to_use_per_rank"]) // self.bytes_per_block
        requested = extra.get("num_cpu_blocks")
        if requested is not None and (
            type(requested) is not int or not 0 < requested <= local_capacity
        ):
            raise ValueError(
                f"num_cpu_blocks={requested!r} must be between 1 and "
                f"local offload capacity {local_capacity}"
            )
        self.num_cpu_blocks = (
            requested if requested is not None else max(1, local_capacity)
        )
        self.backend.register_region("cpu", self.num_cpu_blocks, self.bytes_per_block)

    def begin_step(self, metadata: SimpleCPUOffloadMetadata) -> None:
        assert isinstance(metadata, SimpleCPUOffloadMetadata)
        if metadata.need_flush:
            self.backend.wait_all()
        if metadata.load_event != INVALID_JOB_ID:
            self.backend.submit(
                self.transfer(
                    "load",
                    metadata.load_event,
                    metadata.load_gpu_blocks,
                    metadata.load_cpu_blocks,
                )
            )

    def end_step(self, metadata: SimpleCPUOffloadMetadata) -> KVConnectorOutput:
        if metadata.store_event != INVALID_JOB_ID:
            self.backend.submit(
                self.transfer(
                    "store",
                    metadata.store_event,
                    metadata.store_gpu_blocks,
                    metadata.store_cpu_blocks,
                )
            )
        receiving = set()
        stores = {}
        for result in self.backend.poll():
            job = result.job
            if job.direction == "load":
                receiving.update(metadata.load_event_to_reqs.get(job.job_id, ()))
            else:
                stores[job.job_id] = 1
        return KVConnectorOutput(
            finished_recving=receiving or None,
            kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(stores)
            if stores
            else None,
        )


class OffloadingAdapter(ConnectorAdapter):
    def __init__(self, config, kv_cache_config):
        super().__init__(kv_cache_config)
        spec = cpu_offloading_spec(config, kv_cache_config)
        if spec.num_chunks == 0:
            raise ValueError("Simulator CPU offload capacity is smaller than one chunk")
        self.backend.register_region(
            "cpu", spec.num_chunks, spec.cpu_page_size_per_worker
        )
        self._deferred_stores: list[TransferJob] = []
        self._load_requests: dict[int, str] = {}

    def _transfer(self, direction, job_id, entry) -> TransferJob:
        gpu, cpu = (
            (entry.dst_spec, entry.src_spec)
            if direction == "load"
            else (entry.src_spec, entry.dst_spec)
        )
        assert isinstance(gpu, GPULoadStoreSpec)
        assert isinstance(cpu, CPULoadStoreSpec)
        return self.transfer(direction, job_id, gpu.block_ids, cpu.block_ids)

    def begin_step(self, metadata: OffloadingConnectorMetadata) -> None:
        assert isinstance(metadata, OffloadingConnectorMetadata)
        for job in self._deferred_stores:
            self.backend.submit(job)
        self._deferred_stores.clear()
        flush = metadata.jobs_to_flush or set()
        for job_id in flush:
            entry = metadata.store_jobs.get(job_id)
            if entry is not None:
                self.backend.submit(self._transfer("store", job_id, entry))
        self.backend.wait(
            {("store", job_id) for job_id in flush}
            | {("load", job_id) for job_id in flush}
        )
        sync_loads = set()
        for job_id, entry in metadata.load_jobs.items():
            self.backend.submit(self._transfer("load", job_id, entry))
            if entry.sync:
                sync_loads.add(("load", job_id))
            else:
                self._load_requests[job_id] = entry.req_id
        self.backend.wait(sync_loads)

    def end_step(self, metadata: OffloadingConnectorMetadata) -> KVConnectorOutput:
        self._deferred_stores.extend(
            self._transfer("store", job_id, entry)
            for job_id, entry in metadata.store_jobs.items()
            if job_id not in (metadata.jobs_to_flush or ())
        )
        receiving = set()
        worker_meta = OffloadingWorkerMetadata()
        for result in self.backend.poll():
            job = result.job
            worker_meta.mark_completed(job.job_id)
            getattr(worker_meta.transfer_stats, job.direction).record(
                job.num_bytes, result.duration
            )
            if job.direction == "load":
                request_id = self._load_requests.pop(job.job_id, None)
                if request_id is not None:
                    receiving.add(request_id)
        return KVConnectorOutput(
            finished_recving=receiving or None,
            kv_connector_worker_meta=worker_meta
            if worker_meta.completed_jobs
            else None,
        )

    def shutdown(self) -> None:
        self._deferred_stores.clear()
        self._load_requests.clear()
        super().shutdown()


def create_connector_adapter(config, kv_cache_config) -> ConnectorAdapter | None:
    transfer = config.kv_transfer_config
    if transfer is None:
        return None
    if transfer.kv_connector == "SimpleCPUOffloadConnector":
        if not config.cache_config.enable_prefix_caching:
            return None
        return SimpleCPUOffloadAdapter(config, kv_cache_config)
    if transfer.kv_connector == "OffloadingConnector":
        return OffloadingAdapter(config, kv_cache_config)
    raise ValueError(f"Unsupported simulator connector: {transfer.kv_connector}")
