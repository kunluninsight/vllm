# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ML-trained per-iteration latency predictor.

Build a prefill/decode sklearn bundle from collector exports with
`MLTimePredictor.build_model_from_schedule_batch`. Training and inference
share the simulator's common `ScheduleBatch` feature schema.
"""

import gzip
import json
import math
import os
import warnings
from logging import getLogger

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from vllm_simulator.time_predictor.base import (
    PerRankTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)

logger = getLogger(__name__)


class MLTimePredictor(PerRankTimePredictor):
    """Predict a step latency from batch-composition features."""

    FEATURE_NAMES = [
        "batch_size",
        "sum_extend",
        "max_extend",
        "min_extend",
        "sum_past",
        "max_past",
        "min_past",
        "sum_extend_x_past",
        "sum_extend_squared",
        "sum_past_squared",
        "sum_attn_flops",
        "sum_extend_x_max_past",
        "log1p_sum_past",
        "log1p_sum_attn_flops",
        "batch_size_x_sum_extend",
        "max_past_minus_min_past",
        "is_decode",
        "is_prefill",
    ]

    @staticmethod
    def build_model_from_schedule_batch(
        jsonl_paths: str | list[str],
        output_path: str | None = None,
        *,
        loss: str = "squared_error",
        cross_validate: bool = False,
        balance_windows: bool = False,
        fit_sampling: bool = False,
    ) -> dict:
        """Pool collector exports and train one prefill/decode model bundle.

        Pure decode (all extend lengths equal 1) and prefill/mixed steps are
        fitted separately. If either phase is absent, reuse the other model.
        Forward labels are RPC-1 iter_latency in seconds. Optional sampling
        labels come from RPC-2 sample_tokens_latency.

        Args:
            jsonl_paths: One JSONL path or several; .gz is auto-detected.
            output_path: Optional joblib output; use as database_path.
            loss: "squared_error" to fit the conditional mean or "absolute_error"
                to fit a typical step when the latency distribution has a long tail.
            cross_validate: Select each phase's regressor by mean absolute percentage
                error on held-out steps.
                Hold out input files when multiple exports are supplied, otherwise
                use contiguous folds. Selection never uses target workload labels.
            balance_windows: Give each calibration file equal validation weight
                instead of weighting by its step count.
            fit_sampling: Fit a separate batch-dependent sampling model. Saved
                sampling models replace configured sampling compensation.

        Returns:
            A bundle with phase models, features and a training data summary.

        Raises:
            ValueError: No scheduled steps or invalid lengths/latencies.

        """
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor

        if loss not in ("squared_error", "absolute_error"):
            raise ValueError("loss must be squared_error or absolute_error")
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        features, latencies, groups, sampling = [], [], [], []
        sources = []
        for group, path in enumerate(jsonl_paths):
            source_stats = {"path": str(path), "steps": 0, "empty_steps": 0}
            sources.append(source_stats)
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    requests = row.get("requests") or []
                    if not requests:
                        source_stats["empty_steps"] += 1
                        continue
                    if any(
                        not isinstance(value, int) or value < minimum
                        for req in requests
                        for value, minimum in ((req[1], 1), (req[2], 0))
                    ):
                        raise ValueError(f"Invalid scheduled lengths in {path}")
                    batch = ScheduleBatch(
                        [
                            ScheduleRequest(extend_length=req[1], past_kv_length=req[2])
                            for req in requests
                        ]
                    )
                    latency = float(row["iter_latency"])
                    if not math.isfinite(latency) or latency < 0:
                        raise ValueError(f"Invalid iter_latency in {path}: {latency}")
                    features.append(MLTimePredictor._extract_features(batch))
                    latencies.append(latency)
                    groups.append(group)
                    source_stats["steps"] += 1
                    if fit_sampling:
                        seconds = row.get("sample_tokens_latency")
                        if (
                            not isinstance(seconds, (int, float))
                            or not math.isfinite(seconds)
                            or seconds < 0
                        ):
                            raise ValueError(f"Invalid sample_tokens_latency in {path}")
                        sampling.append(seconds)
        if not features:
            raise ValueError("Cannot train an ML model without scheduled steps")
        X, y = np.asarray(features), np.asarray(latencies)
        summary = MLTimePredictor._summarize_training(X, y, sources)
        is_decode = X[:, -2] == 1
        models, selection = {}, {}
        for name, mask in (("prefill", ~is_decode), ("decode", is_decode)):
            if mask.any():
                if cross_validate:
                    models[name], selection[name] = MLTimePredictor._select_model(
                        X[mask],
                        y[mask],
                        np.asarray(groups)[mask],
                        loss,
                        balance_windows=balance_windows,
                    )
                else:
                    models[name] = HistGradientBoostingRegressor(
                        random_state=0, loss=loss
                    ).fit(X[mask], y[mask])
                logger.info("Trained %s model on %d steps", name, int(mask.sum()))
        for name in ("prefill", "decode"):
            if name not in models:
                logger.warning("No %s steps; reusing the other phase's model", name)
                models[name] = next(iter(models.values()))
        bundle = {
            **models,
            "features": list(MLTimePredictor.FEATURE_NAMES),
            "loss": loss,
            "training_summary": summary,
        }
        if cross_validate:
            bundle["selection"] = selection
        if fit_sampling:
            sample_y = np.asarray(sampling)
            if cross_validate:
                bundle["sampling"], sampling_selection = MLTimePredictor._select_model(
                    X,
                    sample_y,
                    np.asarray(groups),
                    loss,
                    balance_windows=balance_windows,
                    normalizer=y + sample_y,
                )
                bundle["selection"]["sampling"] = sampling_selection
            else:
                bundle["sampling"] = HistGradientBoostingRegressor(
                    random_state=0, loss=loss, min_samples_leaf=5
                ).fit(X, sample_y)
        if output_path:
            joblib.dump(bundle, output_path)
            logger.info("ML model written to %s", output_path)
        return bundle

    @staticmethod
    def _summarize_training(X, y, sources):
        import numpy as np

        def quantiles(values):
            return dict(
                zip(
                    ("min", "p50", "p90", "max"),
                    map(float, np.percentile(values, [0, 50, 90, 100])),
                )
            )

        dimensions = ("batch_size", "scheduled_tokens", "max_past_kv_tokens")

        def bounds(bucket):
            return (
                [0, 0] if bucket < 0 else [2 ** int(bucket), 2 ** (int(bucket) + 1) - 1]
            )

        summary = {
            "steps": len(y),
            "sources": sources,
            "empty_steps": sum(s["empty_steps"] for s in sources),
            "phases": {},
        }
        logger.info(
            "ML training data: %d files, %d scheduled steps, %d empty steps skipped",
            len(sources),
            len(y),
            summary["empty_steps"],
        )
        for phase, mask in (("prefill", X[:, -2] == 0), ("decode", X[:, -2] == 1)):
            count = int(mask.sum())
            if not count:
                summary["phases"][phase] = {"steps": 0}
                continue
            values = X[mask][:, [0, 1, 5]]
            buckets = np.floor(np.log2(np.maximum(values, 1))).astype(int)
            buckets[values == 0] = -1
            cells, counts = np.unique(buckets, axis=0, return_counts=True)
            sparse = counts < 5
            stats = {
                "steps": count,
                "fraction": count / len(y),
                **{name: quantiles(values[:, i]) for i, name in enumerate(dimensions)},
                "latency_ms": quantiles(y[mask] * 1000),
                "coverage": {
                    "bucket_scheme": "powers_of_two",
                    "observed_regions": len(counts),
                    "regions_below_5_samples": int(sparse.sum()),
                    "sparse_examples": [
                        dict(
                            zip(dimensions, (bounds(b) for b in cell)),
                            steps=int(n),
                        )
                        for cell, n in zip(cells[sparse][:5], counts[sparse][:5])
                    ],
                },
            }
            summary["phases"][phase] = stats
            logger.info("ML %s training distribution: %s", phase, json.dumps(stats))
            if sparse.any():
                logger.warning(
                    "ML %s: %d/%d observed shape regions have fewer than 5 samples; "
                    "check training_summary for sparse regions. "
                    "This summary does not establish coverage of unobserved regions "
                    "or prediction accuracy.",
                    phase,
                    int(sparse.sum()),
                    len(counts),
                )
        return summary

    @staticmethod
    def _select_model(X, y, groups, loss, *, balance_windows=False, normalizer=None):
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.linear_model import HuberRegressor
        from sklearn.model_selection import GroupKFold, KFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def candidates():
            models = {
                "baseline": HistGradientBoostingRegressor(random_state=0, loss=loss),
                "absolute": HistGradientBoostingRegressor(
                    random_state=0, loss="absolute_error"
                ),
                "relative_leaf5": HistGradientBoostingRegressor(
                    random_state=0, min_samples_leaf=5
                ),
                "huber": make_pipeline(StandardScaler(), HuberRegressor(max_iter=3000)),
            }
            if balance_windows:
                models.update(
                    huber_physical=make_pipeline(
                        ColumnTransformer(
                            [("shape", StandardScaler(), [0, 1, 4, 7, 8])]
                        ),
                        HuberRegressor(max_iter=3000),
                    ),
                    extra_trees=ExtraTreesRegressor(
                        n_estimators=64, min_samples_leaf=3, random_state=0, n_jobs=1
                    ),
                )
            return models

        denominator = np.maximum(y if normalizer is None else normalizer, 1e-9)

        def fit(name, indices):
            model = candidates()[name]
            kwargs = {}
            if name == "relative_leaf5":
                labels = denominator[indices]
                kwargs["sample_weight"] = (np.median(labels) / labels) ** 2
            return model.fit(X[indices], y[indices], **kwargs)

        indices = np.arange(len(y))
        if len(y) < 5:
            return fit("baseline", indices), {
                "selected": "baseline",
                "cv_error_pct": {},
            }
        unique_groups = np.unique(groups)
        if len(unique_groups) > 1:
            folds = list(
                GroupKFold(n_splits=min(5, len(unique_groups))).split(X, y, groups)
            )
        else:
            folds = list(KFold(n_splits=5).split(X))
        scores = {}
        for name in candidates():
            errors = np.empty(len(y))
            for train, validation in folds:
                prediction = np.maximum(fit(name, train).predict(X[validation]), 0)
                errors[validation] = (
                    np.abs(prediction - y[validation]) / denominator[validation]
                )
            score = (
                np.mean([errors[groups == group].mean() for group in unique_groups])
                if balance_windows
                else errors.mean()
            )
            scores[name] = 100 * float(score)
        selected = min(scores, key=scores.get)
        return fit(selected, indices), {
            "selected": selected,
            "cv_error_pct": scores,
            "folds": len(folds),
            "grouped_by_file": len(unique_groups) > 1,
            "balanced_windows": balance_windows,
        }

    def __init__(
        self,
        database_path: str,
        latency_scale: float = 1.0,
        sample_tokens_base_ms: float = 0.0,
        sample_tokens_lo_us_per_token: float = 0.0,
        sample_tokens_hi_us_per_token: float = 0.0,
        sample_tokens_breakpoint_tokens: int = 0,
    ):
        for name, value in (
            ("latency_scale", latency_scale),
            ("sample_tokens_base_ms", sample_tokens_base_ms),
            ("sample_tokens_lo_us_per_token", sample_tokens_lo_us_per_token),
            ("sample_tokens_hi_us_per_token", sample_tokens_hi_us_per_token),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            type(sample_tokens_breakpoint_tokens) is not int
            or sample_tokens_breakpoint_tokens < 0
        ):
            raise ValueError(
                "sample_tokens_breakpoint_tokens must be a non-negative int"
            )
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"MLTimePredictor database_path not found: {database_path}"
            )

        # A sklearn ensemble can emit one version warning per tree.  Preserve
        # the compatibility signal without flooding a 500-tree simulation log.
        with warnings.catch_warnings(record=True) as version_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            bundle = joblib.load(database_path)
        if version_warnings:
            logger.warning(
                "MLTimePredictor loaded a model produced by another sklearn "
                "version (%d compatibility warnings suppressed)",
                len(version_warnings),
            )
        if isinstance(bundle, dict) and "prefill" in bundle and "decode" in bundle:
            self._prefill_model = bundle["prefill"]
            self._decode_model = bundle["decode"]
            saved_features = bundle.get("features", self.FEATURE_NAMES)
        elif isinstance(bundle, dict) and "model" in bundle:
            self._prefill_model = bundle["model"]
            self._decode_model = bundle["model"]
            saved_features = bundle.get("features", self.FEATURE_NAMES)
        else:
            self._prefill_model = bundle
            self._decode_model = bundle
            saved_features = self.FEATURE_NAMES

        if list(saved_features) != self.FEATURE_NAMES:
            raise ValueError(
                "MLTimePredictor feature schema mismatch: "
                f"saved={list(saved_features)!r}, expected={self.FEATURE_NAMES!r}"
            )

        self._sampling_model = (
            bundle.get("sampling") if isinstance(bundle, dict) else None
        )
        self._latency_scale = float(latency_scale)
        # RPC-2 (sample_tokens) compensation; the trained label covers RPC-1
        # only.  Hinge function of the step's scheduled tokens:
        #
        #   sample_tokens_s = base + a_lo * tt + (a_hi - a_lo) * max(0, tt - s)
        #
        # Kept separate from the label prediction; disabled when all terms
        # are 0.
        self.sample_tokens_base_s = float(sample_tokens_base_ms) / 1e3
        self.sample_tokens_lo_s_per_token = float(sample_tokens_lo_us_per_token) / 1e6
        self.sample_tokens_hi_s_per_token = float(sample_tokens_hi_us_per_token) / 1e6
        self.sample_tokens_breakpoint_tokens = max(
            0, int(sample_tokens_breakpoint_tokens)
        )
        self._sample_tokens_enabled = (
            self.sample_tokens_base_s > 0.0
            or self.sample_tokens_lo_s_per_token > 0.0
            or self.sample_tokens_hi_s_per_token > 0.0
        )
        logger.info(
            "MLTimePredictor loaded from %s (prefill=%s, decode=%s, "
            "n_features=%d, latency_scale=%.4f)",
            database_path,
            type(self._prefill_model).__name__,
            type(self._decode_model).__name__,
            len(saved_features),
            self._latency_scale,
        )
        if self._sampling_model is not None:
            logger.info("MLTimePredictor uses learned batch-dependent sampling latency")
        elif self._sample_tokens_enabled:
            logger.info(
                "MLTimePredictor sample_tokens compensation enabled: "
                "base=%.3fms a_lo=%.4fus/token a_hi=%.4fus/token "
                "breakpoint=%d tokens",
                self.sample_tokens_base_s * 1e3,
                self.sample_tokens_lo_s_per_token * 1e6,
                self.sample_tokens_hi_s_per_token * 1e6,
                self.sample_tokens_breakpoint_tokens,
            )

    @staticmethod
    def _extract_features(batch: ScheduleBatch) -> list[float]:
        exts = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]

        batch_size = len(exts)
        sum_extend = sum(exts)
        sum_past = sum(pasts)
        max_extend = max(exts)
        min_extend = min(exts)
        max_past = max(pasts)
        min_past = min(pasts)
        sum_attn_flops = sum(
            extend * (past + extend / 2) for extend, past in zip(exts, pasts)
        )

        return [
            batch_size,
            sum_extend,
            max_extend,
            min_extend,
            sum_past,
            max_past,
            min_past,
            sum(extend * past for extend, past in zip(exts, pasts)),
            sum(extend * extend for extend in exts),
            sum(past * past for past in pasts),
            sum_attn_flops,
            sum_extend * max_past,
            math.log1p(sum_past),
            math.log1p(sum_attn_flops),
            batch_size * sum_extend,
            max_past - min_past,
            int(all(extend == 1 for extend in exts)),
            int(any(extend > 1 for extend in exts)),
        ]

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        features = self._extract_features(batch)
        is_decode = features[-2] == 1
        model = self._decode_model if is_decode else self._prefill_model
        latency = float(model.predict([features])[0]) * self._latency_scale
        return max(latency, 0.0)

    def predict_step_time(self, batch: ScheduleBatch) -> tuple[float, float]:
        if self._sampling_model is None:
            return super().predict_step_time(batch)
        if batch.is_empty():
            return 0.0, 0.0
        sampling = float(
            self._sampling_model.predict([self._extract_features(batch)])[0]
        )
        return self.predict_infer_time(batch), max(sampling, 0.0)

    def predict_sample_tokens_time(self, total_tokens: int) -> float:
        """Return the RPC-2 (sample_tokens) duration in seconds.

        `total_tokens` is this step's scheduled token count, matching the
        `total_tokens` field written by the GPU-side collection hook.
        Returns 0.0 when the compensation is not configured.

        The calibration was fitted on mixed (chunked-prefill) steps only, so
        applying it to a hypothetical decode-only deployment extrapolates the
        token-independent floor beyond its measured domain.
        """
        if not self._sample_tokens_enabled or total_tokens <= 0:
            return 0.0

        tt = float(total_tokens)
        latency = self.sample_tokens_base_s + self.sample_tokens_lo_s_per_token * tt
        if tt > self.sample_tokens_breakpoint_tokens:
            latency += (
                self.sample_tokens_hi_s_per_token - self.sample_tokens_lo_s_per_token
            ) * (tt - self.sample_tokens_breakpoint_tokens)
        return max(latency, 0.0)
