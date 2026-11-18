# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload records shared by Python benchmark clients."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class GenericRequest:
    prompt: str | None = None
    token_ids: list[int] | None = None
    input_length: int = -1
    output_length: int = -1
    custom_params: dict = field(default_factory=dict)
    sampling_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if (self.prompt is None) == (self.token_ids is None):
            raise ValueError("Provide exactly one of prompt or token_ids")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ValueError("prompt must be text")
        if self.token_ids is not None:
            if not self.token_ids or any(
                type(t) is not int or t < 0 for t in self.token_ids
            ):
                raise ValueError("token_ids must contain non-negative integers")
            if self.input_length == -1:
                self.input_length = len(self.token_ids)


def from_sample_requests(samples) -> list[GenericRequest]:
    requests = []
    for sample in samples:
        if sample.multi_modal_data is not None or sample.lora_request is not None:
            raise ValueError("Multimodal and LoRA samples are not supported")
        kwargs = {
            "input_length": sample.prompt_len,
            "output_length": sample.expected_output_len,
        }
        if isinstance(sample.prompt, str):
            request = GenericRequest(prompt=sample.prompt, **kwargs)
        elif isinstance(sample.prompt, list):
            request = GenericRequest(token_ids=list(sample.prompt), **kwargs)
        else:
            raise ValueError("Dataset samples must contain text or token IDs")
        if sample.request_id is not None:
            request.custom_params["request_id"] = sample.request_id
        if sample.timestamp is not None:
            request.custom_params["created_time"] = sample.timestamp
        requests.append(request)
    return requests
