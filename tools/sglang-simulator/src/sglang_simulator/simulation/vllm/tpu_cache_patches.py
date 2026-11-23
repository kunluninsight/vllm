"""Host-only equivalents of deployed vllm_torchtpu KV-cache & scheduler patches.

CPU simulation strips ``VLLM_PLUGINS=torchtpu`` so it does not activate
TpuPlatform.  These lightweight engine-core patches preserve the native
caching and scheduling semantics that vllm_torchtpu registers in
``patch_registry.py``:

1. ``patch_explicit_cache_checkpoint``: honor explicit
   ``cache_control_params.token_positions`` via ``shared_prefix_boundary`` on
   both ``KVCacheManager.get_computed_blocks`` and
   ``get_computed_blocks_for_connector``.
2. ``patch_same_step_prefix_hits``: disallow prefix-cache hits on blocks
   hashed during the current ``Scheduler.schedule()`` call (matching TPU
   ragged paged attention semantics).
3. ``patch_hybrid_kv_load_failure_recovery``: backport multi-group
   ``Scheduler._update_requests_with_invalid_blocks`` recovery (vLLM #50388).
"""

from __future__ import annotations

from functools import wraps
import os

from sglang_simulator.utils import get_logger

logger = get_logger()

_EXPLICIT_CHECKPOINT_MARKER = "_sim_tpu_explicit_cache_checkpoint"
_SAME_STEP_MARKER = "_sim_tpu_same_step_prefix_hit_patch"
_HYBRID_LOAD_RECOVERY_MARKER = "_sim_tpu_hybrid_kv_load_failure_patch"


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _tpu_cache_patches_active() -> bool:
    """True when running a TPU target simulation (or TPU cache env flags)."""
    if _env_enabled("TPU_MTP_PREFILL_ONLY") or _env_enabled(
        "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP"
    ):
        return True
    try:
        from sglang_simulator.simulation.manager import ConfigManager
        from sglang_simulator.simulation.vllm.tpu_kv_geometry import is_tpu_target

        return is_tpu_target(ConfigManager._get_raw_config())
    except Exception:
        return False


def _explicit_boundary(manager, request) -> int | None:
    if not manager.prefix_cache_lookup_enabled(request):
        return None
    params = getattr(request, "sampling_params", None)
    if params is None or not isinstance(getattr(params, "extra_args", None), dict):
        return None
    cache_control = params.extra_args.get("cache_control_params")
    if not isinstance(cache_control, dict):
        return None
    positions = cache_control.get("token_positions")
    if not isinstance(positions, list):
        return None
    position = max(
        (
            p
            for p in positions
            if type(p) is int and 0 < p <= request.num_prompt_tokens
        ),
        default=0,
    )
    scheduler_block_size = manager.coordinator.scheduler_block_size
    boundary = position // scheduler_block_size * scheduler_block_size
    return boundary or None


def patch_explicit_cache_checkpoint(kv_cache_manager_cls=None) -> None:
    """Wrap KVCacheManager prefix lookups to honor explicit cache checkpoints."""
    if kv_cache_manager_cls is None:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        kv_cache_manager_cls = KVCacheManager

    original = getattr(kv_cache_manager_cls, "get_computed_blocks", None)
    if original is not None and not getattr(original, _EXPLICIT_CHECKPOINT_MARKER, False):

        @wraps(original)
        def get_computed_blocks(self, request):
            result = original(self, request)
            if not _tpu_cache_patches_active():
                return result
            boundary = _explicit_boundary(self, request)
            if boundary is None:
                return result
            blocks, num_computed_tokens, _ = result
            return blocks, num_computed_tokens, boundary

        setattr(get_computed_blocks, _EXPLICIT_CHECKPOINT_MARKER, True)
        kv_cache_manager_cls.get_computed_blocks = get_computed_blocks

    original_connector = getattr(
        kv_cache_manager_cls, "get_computed_blocks_for_connector", None
    )
    if original_connector is not None and not getattr(
        original_connector, _EXPLICIT_CHECKPOINT_MARKER, False
    ):

        @wraps(original_connector)
        def get_computed_blocks_for_connector(self, request):
            result = original_connector(self, request)
            if not _tpu_cache_patches_active():
                return result
            boundary = _explicit_boundary(self, request)
            if boundary is None:
                return result
            blocks, num_computed_tokens, _, hit_diverged = result
            return blocks, num_computed_tokens, boundary, hit_diverged

        setattr(get_computed_blocks_for_connector, _EXPLICIT_CHECKPOINT_MARKER, True)
        kv_cache_manager_cls.get_computed_blocks_for_connector = (
            get_computed_blocks_for_connector
        )


def patch_same_step_prefix_hits(scheduler_cls, block_pool_cls=None) -> None:
    """Keep prefix-cache hits off blocks hashed in the current schedule()."""
    if _env_enabled("HISIM_TPU_DISABLE_SAME_STEP_PREFIX_PATCH"):
        return
    if block_pool_cls is None:
        from vllm.v1.core.block_pool import BlockPool

        block_pool_cls = BlockPool

    def unwritten_block_ids(pool) -> set[int]:
        return pool.__dict__.setdefault("_tpu_unwritten_block_ids", set())

    if not getattr(block_pool_cls.get_cached_block, _SAME_STEP_MARKER, False):
        original_insert = block_pool_cls._insert_block_hash
        original_get_cached = block_pool_cls.get_cached_block

        @wraps(original_insert)
        def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens):
            original_insert(self, block_hash_with_group_id, block, num_tokens)
            # Only track unwritten hashes during an active Scheduler.schedule()
            # call so direct BlockPool unit tests outside schedule() stay intact.
            if (
                _tpu_cache_patches_active()
                and self.__dict__.get("_tpu_in_schedule", 0) > 0
            ):
                unwritten_block_ids(self).add(block.block_id)

        @wraps(original_get_cached)
        def get_cached_block(self, block_hash, kv_cache_group_ids):
            blocks = original_get_cached(self, block_hash, kv_cache_group_ids)
            if not _tpu_cache_patches_active():
                return blocks
            unwritten = self.__dict__.get("_tpu_unwritten_block_ids")
            if (
                blocks
                and unwritten
                and any(block.block_id in unwritten for block in blocks)
            ):
                return None
            return blocks

        setattr(_insert_block_hash, _SAME_STEP_MARKER, True)
        setattr(get_cached_block, _SAME_STEP_MARKER, True)
        block_pool_cls._insert_block_hash = _insert_block_hash
        block_pool_cls.get_cached_block = get_cached_block

    original_schedule = getattr(scheduler_cls, "schedule", None)
    if original_schedule is not None and not getattr(
        original_schedule, _SAME_STEP_MARKER, False
    ):

        @wraps(original_schedule)
        def schedule(self, *args, **kwargs):
            manager = getattr(self, "kv_cache_manager", None)
            pool = getattr(manager, "block_pool", None)
            if pool is not None:
                unwritten_block_ids(pool).clear()
                pool._tpu_in_schedule = pool.__dict__.get("_tpu_in_schedule", 0) + 1
            try:
                return original_schedule(self, *args, **kwargs)
            finally:
                if pool is not None:
                    pool._tpu_in_schedule = max(
                        0, pool.__dict__.get("_tpu_in_schedule", 1) - 1
                    )

        setattr(schedule, _SAME_STEP_MARKER, True)
        scheduler_cls.schedule = schedule
        logger.info(
            "[TPU cache] installed same-step prefix-hit guard on BlockPool/Scheduler"
        )


def patch_hybrid_kv_load_failure_recovery(scheduler_cls) -> None:
    """Backport multi-group KV load failure recovery (vLLM #50388)."""
    if scheduler_cls.__dict__.get(_HYBRID_LOAD_RECOVERY_MARKER, False):
        return
    original_update = getattr(
        scheduler_cls, "_update_requests_with_invalid_blocks", None
    )
    if original_update is None:
        return

    @wraps(original_update)
    def _hybrid_update_requests_with_invalid_blocks(
        self,
        requests,
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ):
        if len(self.kv_cache_config.kv_cache_groups) <= 1:
            return original_update(
                self,
                requests,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=evict_blocks,
            )

        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        null_block_id = self.kv_cache_manager.block_pool.null_block.block_id
        for request in requests:
            req_id = request.request_id
            req_block_ids_per_group = self.kv_cache_manager.get_block_ids(req_id)
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )
            request_invalid_block_ids = {
                block_id
                for req_block_ids in req_block_ids_per_group
                for block_id in req_block_ids
                if block_id != null_block_id and block_id in invalid_block_ids
            }
            if request_invalid_block_ids:
                total_affected_tokens += req_num_computed_tokens
                request.num_computed_tokens = 0
                if evict_blocks:
                    for req_block_ids in req_block_ids_per_group:
                        blocks_to_evict.update(
                            block_id
                            for block_id in req_block_ids
                            if block_id != null_block_id
                        )
                affected_req_ids.add(req_id)
        return affected_req_ids, total_affected_tokens, blocks_to_evict

    scheduler_cls._update_requests_with_invalid_blocks = (
        _hybrid_update_requests_with_invalid_blocks
    )
    setattr(scheduler_cls, _HYBRID_LOAD_RECOVERY_MARKER, True)
    logger.info(
        "[TPU cache] installed hybrid KV load failure recovery on Scheduler"
    )


def install_tpu_cache_patches(scheduler_cls) -> None:
    """Install all host-side TPU KV-cache & scheduler patches."""
    patch_explicit_cache_checkpoint()
    patch_same_step_prefix_hits(scheduler_cls)
    patch_hybrid_kv_load_failure_recovery(scheduler_cls)
    if _tpu_cache_patches_active() and not _env_enabled(
        "DS_LLM_ENABLE_PREDICT_CACHE_HINT"
    ):
        logger.warning(
            "[TPU cache] explicit cache checkpoint patch is active, but "
            "DS_LLM_ENABLE_PREDICT_CACHE_HINT is not enabled; DashLLM will not "
            "forward x-ds-max-user-theoretical-hit-num into "
            "cache_control_params.token_positions unless set to 1"
        )
