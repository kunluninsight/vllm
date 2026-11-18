# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implement the worker contract using cache accounting and synthetic tokens."""

import json
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm_simulator.plugin.config import SimulatorConfig
from vllm_simulator.plugin.model_runner import SimulatorModelRunner, TimedModelOutput
from vllm_simulator.time_predictor import ScheduleBatch, ScheduleRequest

logger = init_logger(__name__)


class SimulatorWorker(WorkerBase):
    def init_device(self) -> None:
        self.device = torch.device("cpu")
        self.current_platform.configure_device(
            SimulatorConfig.from_config(self.vllm_config).device
        )
        self.get_kv_cache_layer_configs()
        self.simulator_config = SimulatorConfig.from_config(self.vllm_config)
        self.model_runner = SimulatorModelRunner(self.vllm_config)
        self.step_trace = None
        self.step_index = 0
        if self.simulator_config.step_trace_dir is not None:
            directory = Path(self.simulator_config.step_trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.step_trace = (directory / "dp0.steps.jsonl").open("x", buffering=1)
        self.model_runner.model_memory_usage = self.model_layout.model_memory_bytes
        self.request_lengths: dict[str, tuple[int, int]] = {}

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        pass

    def get_supported_tasks(self):
        return ("generate",)

    def get_kv_connector_handshake_metadata(self):
        return None

    def reset_encoder_cache(self) -> None:
        """The simulator supports decoder-only models and has no encoder cache."""

    def get_kv_cache_layer_configs(self):
        if not hasattr(self, "_kv_cache_layer_configs"):
            from vllm_simulator.plugin.model_layout import build_model_layout

            self.model_layout = build_model_layout(self.vllm_config)
            self._kv_cache_layer_configs = self.model_layout.layer_configs
        return self._kv_cache_layer_configs

    def get_supported_kv_cache_layouts(self):
        self.get_kv_cache_layer_configs()
        return list(self.model_layout.supported_kv_cache_layouts)

    def determine_available_memory(self) -> int:
        if self.cache_config.kv_cache_memory_bytes is not None:
            return self.cache_config.kv_cache_memory_bytes
        device = self.simulator_config.device
        requested = int(
            device.total_memory_bytes
            * self.cache_config.resolved_gpu_memory_utilization
        )
        available = requested - self.model_layout.model_memory_bytes
        if available <= 0:
            raise ValueError(
                f"No KV cache memory remains on {device.name}: "
                f"requested {requested} bytes, "
                f"model tensors {self.model_layout.model_memory_bytes} bytes. "
                "Increase simulated device memory or TP/PP parallelism."
            )
        logger.info(
            "Simulator %s: %.2f GB device memory, %.2f GB model tensors, "
            "%.2f GB available for KV cache "
            "(utilization %.2f)",
            device.name,
            device.total_memory_gb,
            self.model_layout.model_memory_bytes / 1_000_000_000,
            available / 1_000_000_000,
            self.cache_config.resolved_gpu_memory_utilization,
        )
        return available

    def initialize_from_config(self, kv_cache_config) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(0.0, 0.0)

    def execute_model(self, output) -> ModelRunnerOutput | TimedModelOutput:
        for request_id in output.finished_req_ids:
            self.request_lengths.pop(request_id, None)
        computed_tokens = dict(
            zip(
                output.scheduled_cached_reqs.req_ids,
                output.scheduled_cached_reqs.num_computed_tokens,
            )
        )
        for request in output.scheduled_new_reqs:
            self.request_lengths[request.req_id] = (
                request.prompt_len,
                len(request.prefill_token_ids),
            )
            computed_tokens[request.req_id] = request.num_computed_tokens
        ids = list(output.num_scheduled_tokens)
        batch = ScheduleBatch()
        sampled = []
        for rid in ids:
            prompt_length, prefill_length = self.request_lengths[rid]
            num_tokens = output.num_scheduled_tokens[rid]
            computed = computed_tokens[rid]
            batch.reqs.append(
                ScheduleRequest(num_tokens, computed, prompt_length, prefill_length)
            )
            ready = computed + num_tokens >= prefill_length
            sampled.append([self.simulator_config.token_id] if ready else [])
        if ids:
            self.model_runner.forward(batch)
            if self.step_trace is not None:
                self.step_trace.write(
                    json.dumps(
                        {
                            "step_id": self.step_index,
                            "scheduled_at": self.model_runner.submitted_at,
                            "start": self.model_runner.last_step_start,
                            "end": self.model_runner.clock.device_available_at,
                            "requests": [
                                [rid, req.extend_length, req.past_kv_length]
                                for rid, req in zip(ids, batch.reqs)
                            ],
                        }
                    )
                    + "\n"
                )
                self.step_index += 1
        result = ModelRunnerOutput(req_ids=ids, sampled_token_ids=sampled)
        if ids:
            return TimedModelOutput(
                result,
                self.model_runner.clock,
                self.model_runner.clock.device_available_at,
            )
        return result

    def shutdown(self) -> None:
        if self.step_trace is not None:
            self.step_trace.close()
