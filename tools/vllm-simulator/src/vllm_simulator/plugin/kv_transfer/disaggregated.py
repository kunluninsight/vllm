# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native scheduler handoffs between independent BLOCKING vLLM servers."""

import hashlib
import json
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus
from vllm_simulator.plugin.kv_transfer.client import BrokerClient, BrokerError
from vllm_simulator.plugin.kv_transfer.layout import PDKVLayout
from vllm_simulator.plugin.kv_transfer.protocol import CONNECTOR_MODULE, CONNECTOR_NAME

logger = init_logger(__name__)
__all__ = ["CONNECTOR_MODULE", "CONNECTOR_NAME", "SimulatorPDConnector"]


def prompt_hash(token_ids):
    return hashlib.sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class PendingReceive:
    started_at: float
    local_tokens: int


@dataclass
class PDConnectorMetadata(KVConnectorMetadata):
    sending: set[str] = field(default_factory=set)
    receiving: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)


class SimulatorPDConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        transfer = vllm_config.kv_transfer_config
        extra = transfer.kv_connector_extra_config
        self.producer = transfer.kv_role == "kv_producer"
        self.owner = f"{transfer.engine_id}:{uuid.uuid4().hex}"
        self.layout = PDKVLayout.from_config(
            vllm_config, kv_cache_config, self.owner
        ).to_wire()
        self.client = BrokerClient(
            extra["broker_url"], self.owner, extra.get("rpc_timeout_s", 2)
        )
        self.poll_interval = extra.get("poll_interval_s", 0.005)
        self._sending: dict[str, str] = {}
        self._receiving: dict[str, str] = {}
        self._loaded: set[str] = set()
        self._matches: dict[str, PendingReceive] = {}
        self._failures: set[str] = set()
        self._stats: dict[str, dict] = {}
        self.publish_errors: dict[str, str] = {}
        self.client.poll()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if (
            self.producer
            or not params.get("do_remote_prefill")
            or request.request_id in self._loaded
        ):
            return 0, False
        count = max(0, request.num_prompt_tokens - num_computed_tokens)
        if count:
            pending = self._matches.get(request.request_id)
            if pending is None:
                self._matches[request.request_id] = PendingReceive(
                    time.monotonic(), num_computed_tokens
                )
            else:
                pending.local_tokens = num_computed_tokens
        return count, bool(count)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if not num_external_tokens:
            return
        params = request.kv_transfer_params
        rid = request.request_id
        self._loaded.add(rid)
        pending = self._matches.pop(rid)
        destination_wait = time.monotonic() - pending.started_at
        try:
            result = self.client.call(
                "receive",
                owner=self.owner,
                request_id=rid,
                transfer_id=params.get("transfer_id"),
                source=params["remote_engine_id"],
                source_request=params["remote_request_id"],
                num_tokens=request.num_prompt_tokens,
                layout=asdict(self.layout),
                prompt_hash=prompt_hash(request.prompt_token_ids),
                local_tokens=pending.local_tokens,
                blocks=blocks.get_block_ids()[0],
                destination_wait_s=destination_wait,
            )
            params["transfer_id"] = result["transfer_id"]
        except BrokerError as error:
            logger.warning("KV handoff allocation failed: %s", error)
            self._failures.add(rid)
        self._receiving[rid] = params.get("transfer_id", "")
        params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output):
        metadata = PDConnectorMetadata()
        if self._sending or self._receiving:
            wait = (
                self.poll_interval
                if not scheduler_output.total_num_scheduled_tokens
                else 0
            )
            try:
                previous_broker = self.client.broker_id
                events = self.client.poll(wait)
                if previous_broker != self.client.broker_id:
                    # New offers may already have been published or claimed on
                    # the replacement broker before this scheduler step.
                    prefix = f"{self.client.broker_id}:"
                    metadata.sending.update(
                        rid
                        for rid, token in self._sending.items()
                        if not token.startswith(prefix)
                    )
                    self._failures.update(
                        rid
                        for rid, token in self._receiving.items()
                        if not token.startswith(prefix)
                    )
                for event in events:
                    rid = event["request_id"]
                    token = event["transfer_id"]
                    if event["kind"] == "send" and self._sending.get(rid) == token:
                        metadata.sending.add(rid)
                    elif event["kind"] == "recv" and self._receiving.get(rid) == token:
                        metadata.receiving.add(rid)
                        if event["failed"]:
                            metadata.failed.add(rid)
                        else:
                            self._stats[rid] = event["stats"]
            except BrokerError as error:
                logger.warning("Retrying KV broker poll after error: %s", error)
        metadata.failed.update(self._failures & self._receiving.keys())
        metadata.receiving.update(metadata.failed)
        return metadata

    def request_finished(self, request, block_ids):
        rid = request.request_id
        params = request.kv_transfer_params or {}
        self._loaded.discard(rid)
        self._matches.pop(rid, None)
        if (
            self.producer
            and params.get("do_remote_decode")
            and request.status
            in (
                RequestStatus.FINISHED_LENGTH_CAPPED,
                RequestStatus.FINISHED_STOPPED,
            )
        ):
            try:
                result = self.client.call(
                    "publish",
                    owner=self.owner,
                    request_id=rid,
                    layout=asdict(self.layout),
                    prompt_hash=prompt_hash(request.prompt_token_ids),
                    num_tokens=request.num_prompt_tokens,
                    blocks=block_ids,
                )
            except BrokerError as error:
                logger.warning("KV handoff publication failed: %s", error)
                self.publish_errors[rid] = str(error)
                return False, None
            self._sending[rid] = result["transfer_id"]
            return True, {
                "do_remote_prefill": True,
                "do_remote_decode": False,
                "remote_engine_id": self.owner,
                "remote_request_id": rid,
                "remote_block_ids": block_ids,
                "remote_num_tokens": request.num_prompt_tokens,
                "tp_size": self.layout.tp_size,
                "dcp_size": 1,
                "pp_size": 1,
                "transfer_id": result["transfer_id"],
            }
        if not self.producer and (
            rid in self._receiving or params.get("do_remote_prefill")
        ):
            try:
                self.client.call(
                    "cancel",
                    owner=self.owner,
                    request_id=rid,
                    transfer_id=params.get("transfer_id"),
                    source=params.get("remote_engine_id"),
                    source_request=params.get("remote_request_id"),
                )
            except BrokerError as error:
                logger.warning("KV handoff cancellation failed: %s", error)
            return rid in self._receiving, None
        stats = self._stats.pop(rid, None)
        return False, {"simulation_kv_transfer": stats} if stats is not None else None

    def update_connector_output(self, connector_output: KVConnectorOutput):
        for rid in connector_output.finished_sending or ():
            self._sending.pop(rid, None)
        for rid in connector_output.finished_recving or ():
            self._receiving.pop(rid, None)
            self._failures.discard(rid)

    def has_pending_block_frees(self):
        return bool(self._sending)

    def has_pending_push_work(self):
        return bool(self._sending)

    def reset_cache(self):
        return not (self._sending or self._receiving)

    def shutdown(self):
        with suppress(BrokerError):
            self.client.call("close", owner=self.owner)

    def start_load_kv(self, forward_context, **kwargs):
        raise RuntimeError("Simulator PD uses the metadata worker adapter")

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass
