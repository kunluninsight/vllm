# vLLM simulator plugin

Run vLLM serving or virtual-time workload replay on a single CPU host using
synthetic tokens and predicted model latency.

- [Installation, serving and replay](../../docs/benchmarking/simulator.md)
- [Collect native timings and build a predictor](docs/collectors.md)

## Run the end-to-end tests

From the repository root with vLLM and the plugin installed, run

```bash
CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS=simulator \
  .venv/bin/python -m pytest --confcutdir=tools/vllm-simulator tools/vllm-simulator/tests
```

Tests use public model configurations. For offline use, set
`SIMULATOR_TEST_MODEL_DIR` to a directory containing the same
`organization/model` subdirectories. The HTTP serving test also needs the
Qwen3-4B tokenizer.
