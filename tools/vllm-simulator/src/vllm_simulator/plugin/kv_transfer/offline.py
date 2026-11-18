# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import an always-available prompt KV snapshot into a virtual-time engine."""

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

CONNECTOR_NAME = "SimulatorOfflineKVConnector"
CONNECTOR_MODULE = "vllm_simulator.plugin.kv_transfer.offline"


def validate_import(params, prompt_length):
    count = params.get("offline_kv_tokens", 0)
    if type(count) is not int or not 0 <= count <= prompt_length:
        raise ValueError(
            "offline_kv_tokens must be an integer within the prompt length"
        )
    if params.get("do_remote_prefill") or params.get("do_remote_decode"):
        raise ValueError("Offline KV import cannot be combined with online P/D handoff")
    return count


@dataclass
class OfflineKVMetadata(KVConnectorMetadata):
    receiving: set[str]


class SimulatorOfflineKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        groups = kv_cache_config.kv_cache_groups
        if (
            len(groups) != 1
            or groups[0].manager_config.manager_cls is not FullAttentionManager
        ):
            raise ValueError("Offline KV import requires one full-attention KV group")
        self._pending: set[str] = set()
        self._loaded: dict[str, int] = {}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if self._loaded.get(request.request_id) == request.num_preemptions:
            return 0, False
        count = validate_import(
            request.kv_transfer_params or {}, request.num_prompt_tokens
        )
        # The only prompt token must run to sample; importing it saves no work.
        if request.num_tokens == 1:
            return 0, False
        missing = max(0, count - num_computed_tokens)
        return missing, bool(missing)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            self._pending.add(request.request_id)
            self._loaded[request.request_id] = request.num_preemptions

    def build_connector_meta(self, scheduler_output):
        return OfflineKVMetadata(self._pending.copy())

    def update_connector_output(self, connector_output):
        self._pending.difference_update(connector_output.finished_recving or ())

    def request_finished(self, request, block_ids):
        self._loaded.pop(request.request_id, None)
        return request.request_id in self._pending, None

    def reset_cache(self):
        return not self._pending

    def start_load_kv(self, forward_context, **kwargs):
        raise RuntimeError("Offline KV import uses the metadata worker adapter")

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass
