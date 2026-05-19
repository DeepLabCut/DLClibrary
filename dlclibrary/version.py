# Backwards compatibility shim only. Consider deprecated.
import warnings

from . import __version__ as _PACKAGE_VERSION
from . import VERSION as _PACKAGE_VERSION_ALIAS

__all__ = ["__version__", "VERSION"]


def __getattr__(name):
    if name == "__version__":
        warnings.warn(
            (
                "'dlclibrary.version.__version__' is deprecated and will be removed "
                "in a future release. Use 'dlclibrary.__version__' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return _PACKAGE_VERSION

    if name == "VERSION":
        warnings.warn(
            (
                "'dlclibrary.version.VERSION' is deprecated and will be removed "
                "in a future release. Use 'dlclibrary.VERSION' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return _PACKAGE_VERSION_ALIAS

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
