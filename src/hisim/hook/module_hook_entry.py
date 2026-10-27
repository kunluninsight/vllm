import importlib.abc
import re
import sys
from typing import List, Union

from hisim.hook.base_hook import BaseHook, _register_hooks
from hisim.utils import get_logger

logger = get_logger("sgl_simulator")

MODULE_HOOKS: List[BaseHook] = []


def _matches(hook, fullname):
    name = hook.HOOK_MODULE_NAME
    return name is not None and (
        bool(re.search(name, fullname)) if hook.REGEX else name == fullname
    )


class HookMetaPathFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not any(_matches(hook, fullname) for hook in MODULE_HOOKS):
            return None
        # Delegate discovery to the remaining finders so the original loader,
        # package metadata, __file__, and source/extension semantics survive.
        index = sys.meta_path.index(self)
        for finder in sys.meta_path[index + 1 :]:
            if isinstance(finder, HookMetaPathFinder):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = HookLoader(spec.loader)
                return spec
        return None


class HookLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        for hook in MODULE_HOOKS:
            if _matches(hook, module.__name__):
                logger.debug(
                    "Hooking Module: %s into %s", hook.__name__, module.__name__
                )
                try:
                    hook.hook(module)
                except Exception as exc:
                    logger.warning(
                        "Failed to hook module [%s]. Error: %s", module.__name__, exc
                    )


def install_module_hooks(hooks: Union[List[BaseHook], BaseHook]):
    _register_hooks(MODULE_HOOKS, hooks)
    if not any(isinstance(finder, HookMetaPathFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, HookMetaPathFinder())


def remove_module_hooks():
    # Clear the registered hooks and reset the module finder.
    # Note: The modules that have been hooked will not be reset.
    MODULE_HOOKS.clear()
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, HookMetaPathFinder)
    ]
