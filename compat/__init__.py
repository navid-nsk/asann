# compat — Backward-compatibility shims for the csann -> asann rename.
#
# This package is imported automatically by asann/__init__.py.
# It installs a meta-path finder that intercepts "import csann.*" and
# redirects to "asann.*", so that torch.load() can unpickle models
# saved under the old package name.

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys


class _AliasImporter(importlib.abc.MetaPathFinder):
    """Meta-path finder that redirects old_prefix.* -> new_prefix.*"""

    def __init__(self, old_prefix: str, new_prefix: str):
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix

    def find_spec(self, fullname, path, target=None):
        if fullname != self.old_prefix and not fullname.startswith(self.old_prefix + "."):
            return None
        # Compute the real module name
        if fullname == self.old_prefix:
            real_name = self.new_prefix
        else:
            real_name = self.new_prefix + fullname[len(self.old_prefix):]
        return importlib.machinery.ModuleSpec(
            fullname,
            _AliasLoader(real_name),
        )


class _AliasLoader(importlib.abc.Loader):
    """Loader that imports the real module and registers it under the alias."""

    def __init__(self, real_name: str):
        self.real_name = real_name

    def create_module(self, spec):
        return None  # use default semantics

    def exec_module(self, module):
        real_mod = importlib.import_module(self.real_name)
        # Replace the module's contents with the real module
        module.__dict__.update(real_mod.__dict__)
        module.__path__ = getattr(real_mod, "__path__", [])
        module.__loader__ = self
        # Also register so future imports skip the finder
        sys.modules[module.__name__] = real_mod


_installed = False


def activate() -> None:
    """Install backward-compatibility import hooks.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _installed
    if _installed:
        return
    _installed = True

    # csann -> asann (always available since we are called from asann.__init__)
    sys.meta_path.insert(0, _AliasImporter("csann", "asann"))

    # csann_cuda -> asann_cuda (only if asann_cuda is available)
    try:
        import asann_cuda  # noqa: F401
        sys.meta_path.insert(0, _AliasImporter("csann_cuda", "asann_cuda"))
    except ImportError:
        pass  # CUDA ops not built — nothing to alias
