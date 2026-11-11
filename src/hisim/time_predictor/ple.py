"""Remote PLE admission latency for Qwen4-style per-layer n-gram embeddings.

On a remote_v6d Prefill instance, HybridConnector fetches PLE rows for the
whole prompt after external KV load and before the request is schedulable:
hash -> remote Prepare/Fetch -> local attach. Per-step PLE compute is part of
the forward and is left to the step predictor.
"""

import math
from dataclasses import dataclass

# Calibrated on L20B against the remote PLE service (Qwen4, 16 heads x 512
# FP8 per token), p50 over T in [1K, 1M], R^2 = 0.997.
DEFAULT_BASE_MS = 5.56
DEFAULT_PER_1K_TOKENS_MS = 0.383


def model_has_ple(hf_text_config) -> bool:
    """Match Qwen4Exp decoder layers: PLE needs use_ple and nonempty ple_layer_ids."""
    layer_ids = getattr(hf_text_config, "ple_layer_ids", None)
    return bool(getattr(hf_text_config, "use_ple", bool(layer_ids)) and layer_ids)


@dataclass(frozen=True)
class PLEAdmissionModel:
    enabled: bool = True
    base_ms: float = DEFAULT_BASE_MS
    per_1k_tokens_ms: float = DEFAULT_PER_1K_TOKENS_MS

    def __post_init__(self):
        for name in ("base_ms", "per_1k_tokens_ms"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"ple.admission.{name} must be a finite number >= 0")

    @classmethod
    def from_config(cls, config: dict | None) -> "PLEAdmissionModel":
        """Build from the hisim config ``ple.admission`` section."""
        config = config or {}
        unknown = set(config) - {"enabled", "base_ms", "per_1k_tokens_ms"}
        if unknown:
            raise ValueError(f"Unknown ple.admission keys: {sorted(unknown)}")
        return cls(
            enabled=bool(config.get("enabled", True)),
            base_ms=config.get("base_ms", DEFAULT_BASE_MS),
            per_1k_tokens_ms=config.get("per_1k_tokens_ms", DEFAULT_PER_1K_TOKENS_MS),
        )

    def latency(self, num_prompt_tokens: int) -> float:
        """Seconds to prepare the full prompt; prefix hits do not reduce it."""
        if not self.enabled or num_prompt_tokens <= 0:
            return 0.0
        return (self.base_ms + self.per_1k_tokens_ms * num_prompt_tokens / 1024) / 1000
