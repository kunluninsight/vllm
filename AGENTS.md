## Coding
- Keep comments concise; add detail only for non-obvious behavior.
- Hooks under `src/hisim/simulation/hijack/` subclass `hisim.hook.BaseHook`; name class hooks `C_<Name>Hook` and module hooks `M_<Name>Hook`.
  - Register every new hook in the nearest package `__init__.py` via `get_class_hooks()` / `get_module_hooks()`; nested packages (e.g. `vllm/v6d`) are aggregated by their parent.
  - `hijack/startup.py` installs hooks only from the top-level `vllm`, `dashllm`, and `v6d` packages; when adding a new top-level package, also register it there.
- Reserve `test/` for E2E validation only. Keep local unit tests in `tmp/ut` and never commit them.
  - If a new feature's metrics are observable via E2E, extend the existing E2E test 
  (by adding startup args and result assertions) once it has been validated by separate unit tests.
- Run `ruff format` on all changed Python files before finishing.
- New model adaptation:
  - Check `src/hisim/simulation/hijack/vllm/kv_layout.py` first: every simulated KV cache group (attention and state specs such as Mamba) and its page size must match the real engine.

## Testing
- This project targets PAI-vLLM; run E2E tests inside the serving image.
- Once v6d and a local model are available, run:
  `python test/test_dashllm_single_node.py --model <local-model-path>`
  and ensure it passes.
- On upstream vLLM (no `dashservingd` / `v6d` on PATH), skip this test and report it as **skipped**, not failed.
- If Hugging Face downloads fail, retry with `HF_ENDPOINT=https://hf-mirror.com`.

## Environment
- At session start, read `tmp/env.local.md` once if it exists; it contains deployment-specific configuration that overrides all defaults.