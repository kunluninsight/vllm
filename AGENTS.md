## Coding
- Keep comments concise; add detail only for complex behavior.
- In `hijack`, prefix class hooks with `C_` and module hooks with `M_`.
- Reserve `test/` for end-to-end (E2E) validation. Keep local unit tests in `tmp/ut` without committing them.
- Format changed Python files with `ruff format` before finishing.


## Testing
- If Hugging Face model downloads fail, retry with `HF_ENDPOINT=https://hf-mirror.com`.
- This project targets PAI-vLLM; run E2E tests inside the serving image.
- Once v6d and a local model are available, run `python test/test_dashllm_single_node.py --model <local-model-path>` and ensure it passes.
- On upstream vLLM (no `dashservingd`/`v6d` on PATH), skip this test and report it as skipped rather than failed.
