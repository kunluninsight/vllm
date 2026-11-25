"""Install simulator hooks before importing the inference frameworks."""

import logging
import os
import re
import sys

# Explicit entrypoints and the .pth bootstrap can share the same interpreter.
_HOOKS_INSTALLED = False


def init_hook():
    """Install hooks once before framework imports; the bootstrap gates activation."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return True

    # The framework-owned controls stay in startup, not simulator env.py.
    os.environ.setdefault("SRPC_STREAM_DISABLE_RDMA", "1")
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "42"
        logging.getLogger("hisim").warning(
            "[init_hook] PYTHONHASHSEED not set, defaulting to 42 for cross-node hash consistency"
        )

    # Runtime torch/Triton patches must precede any framework import.
    from hisim.simulation.hijack import torch as torch_hijack

    torch_hijack.patch_runtime()

    import hisim.hook as hisim_hook
    from hisim.simulation.hijack import dashllm, v6d, vllm

    class_hooks = [
        *torch_hijack.get_class_hooks(),
        *vllm.get_class_hooks(),
        *dashllm.get_class_hooks(),
        *v6d.get_class_hooks(),
    ]
    module_hooks = [
        *torch_hijack.get_module_hooks(),
        *vllm.get_module_hooks(),
        *dashllm.get_module_hooks(),
        *v6d.get_module_hooks(),
    ]
    class_hooks = list(dict.fromkeys(class_hooks))
    module_hooks = list(dict.fromkeys(module_hooks))
    if class_hooks:
        hisim_hook.install_class_hooks(class_hooks)
    if module_hooks:
        hisim_hook.install_module_hooks(module_hooks)

    # Patch already-loaded classes and modules; the finder handles future imports.
    logger = logging.getLogger("hisim")
    for hook in class_hooks:
        mod_pattern = getattr(hook, "HOOK_MODULE_NAME", None)
        cls_name = getattr(hook, "HOOK_CLASS_NAME", None)
        if not mod_pattern or not cls_name:
            continue
        for name, module in tuple(sys.modules.items()):
            if module is None:
                continue
            matches = (
                bool(re.search(mod_pattern, name))
                if getattr(hook, "REGEX", False)
                else name == mod_pattern
            )
            if matches:
                target_cls = getattr(module, cls_name, None)
                if target_cls is not None:
                    try:
                        hook.hook(target_cls)
                    except Exception as exc:
                        logger.warning(
                            "Failed to hook loaded class [%s.%s]. Error: %s",
                            name,
                            cls_name,
                            exc,
                        )
    for hook in module_hooks:
        for name, module in tuple(sys.modules.items()):
            if module is None:
                continue
            matches = (
                bool(re.search(hook.HOOK_MODULE_NAME, name))
                if hook.REGEX
                else name == hook.HOOK_MODULE_NAME
            )
            if matches:
                try:
                    hook.hook(module)
                except Exception as exc:
                    logger.warning("Failed to hook module [%s]. Error: %s", name, exc)

    _HOOKS_INSTALLED = True
    logger.info(
        "[init_hook] Installed %d class hooks and %d module hooks",
        len(class_hooks),
        len(module_hooks),
    )
    return True
