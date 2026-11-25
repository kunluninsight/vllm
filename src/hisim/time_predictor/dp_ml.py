"""DP-aware latency prediction for measured active-worker execution spans."""

import math
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import ThreadpoolController

from hisim.time_predictor.base import InferTimePredictor, ScheduleBatch, ScheduleRequest
from hisim.time_predictor.ml import MLTimePredictor
from hisim.utils import get_logger
from hisim.utils.download import resolve_local_path

logger = get_logger()


class DPMLTimePredictor(InferTimePredictor):
    FEATURE_SCHEMA = "dp_v1"
    LABEL_KIND = "max_active_worker_execute_model_cuda_events_s"
    SUMMARY_FEATURES = (
        "active_ranks",
        "idle_ranks",
        "max_rank_tokens",
        "min_rank_tokens",
        "max_rank_batch",
        "min_rank_batch",
        "token_imbalance",
    )

    @classmethod
    def feature_names(cls, dp_size):
        names = [f"global_{name}" for name in MLTimePredictor.FEATURE_NAMES]
        names.extend(cls.SUMMARY_FEATURES)
        for rank in range(dp_size):
            names.extend(
                f"load_rank_{rank}_{name}" for name in MLTimePredictor.FEATURE_NAMES
            )
        return names

    @classmethod
    def extract_features(cls, requests_by_dp):
        if not requests_by_dp:
            raise ValueError("DP features require at least one rank")
        batches = [
            ScheduleBatch([ScheduleRequest(int(r[0]), int(r[1])) for r in local])
            for local in requests_by_dp
        ]
        if any(
            r.extend_length < 0 or r.past_kv_length < 0 for b in batches for r in b.reqs
        ):
            raise ValueError("DP request lengths must be nonnegative")
        flat = ScheduleBatch([r for batch in batches for r in batch.reqs])
        features = MLTimePredictor._extract_features(flat) if flat.reqs else [0.0] * 18
        ranks = [
            MLTimePredictor._extract_features(b) if b.reqs else [0.0] * 18
            for b in batches
        ]
        tokens = [f[1] for f in ranks]
        sizes = [f[0] for f in ranks]
        active = sum(bool(batch.reqs) for batch in batches)
        features.extend(
            [
                active,
                len(batches) - active,
                max(tokens),
                min(tokens),
                max(sizes),
                min(sizes),
                max(tokens) / sum(tokens) if sum(tokens) else 0.0,
            ]
        )
        # Rank identities are arbitrary; preserve load composition with a canonical order.
        ranks.sort(key=lambda f: (f[1], f[0], *f[2:]), reverse=True)
        features.extend(value for rank in ranks for value in rank)
        return features

    def __init__(
        self,
        model,
        hw,
        config,
        database_path,
        latency_scale=1.0,
        sample_tokens_base_ms=0.0,
    ):
        super().__init__(model, hw, config)
        with warnings.catch_warnings(record=True) as version_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            bundle = joblib.load(resolve_local_path(database_path))
        if version_warnings:
            logger.warning(
                "DPMLTimePredictor bundle has %d sklearn version warnings",
                len(version_warnings),
            )
        if not isinstance(bundle, dict):
            raise TypeError("DP ML bundle must be a dictionary")
        if bundle.get("feature_schema") != self.FEATURE_SCHEMA:
            raise ValueError("DP ML bundle requires feature_schema=dp_v1")
        self.dp_size = bundle["dp_size"]
        if type(self.dp_size) is not int or self.dp_size < 1:
            raise ValueError("DP ML bundle has invalid dp_size")
        if bundle.get("features") != self.feature_names(self.dp_size):
            raise ValueError("DP ML feature schema mismatch")
        if bundle.get("label_kind") != self.LABEL_KIND:
            raise ValueError("DP ML label kind mismatch")
        self._model = bundle["model"]
        # Single-row inference needs no tree pool; keep both DP ranks deterministic.
        for estimator in (self._model, getattr(self._model, "regressor_", None)):
            if hasattr(estimator, "n_jobs"):
                estimator.n_jobs = 1
        if self._model.n_features_in_ != len(bundle["features"]):
            raise ValueError("DP ML estimator feature count mismatch")
        self._scale = float(latency_scale)
        self._sample_s = float(sample_tokens_base_ms) / 1000
        if (
            not math.isfinite(self._scale)
            or self._scale <= 0
            or not math.isfinite(self._sample_s)
            or self._sample_s < 0
        ):
            raise ValueError(
                "DP ML scale must be positive and sampling cost nonnegative"
            )
        self._threadpool = ThreadpoolController()
        logger.info(
            "DPMLTimePredictor loaded %s (dp=%d, features=%d, label=%s)",
            database_path,
            self.dp_size,
            len(bundle["features"]),
            self.LABEL_KIND,
        )

    def predict_dp_infer_time(self, requests_by_dp):
        if len(requests_by_dp) != self.dp_size:
            raise ValueError(
                f"DP ML expects {self.dp_size} ranks, got {len(requests_by_dp)}"
            )
        if not any(requests_by_dp):
            return 0.0
        features = self.extract_features(requests_by_dp)
        with self._threadpool.limit(limits=1, user_api="openmp"):
            value = float(self._model.predict([features])[0]) * self._scale
        if not math.isfinite(value):
            raise ValueError("DP ML returned non-finite latency")
        return max(value, 0.0)

    def predict_dp_sample_tokens_time(self, requests_by_dp):
        return self._sample_s if any(requests_by_dp) else 0.0

    def predict_infer_time(self, batch):
        raise ValueError(
            "DP ML requires rank-separated input via predict_dp_infer_time"
        )
