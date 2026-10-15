#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Centralized helpers for reading, writing, and creating configuration files (YAML)."""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

import numpy as np
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


def get_yaml_loader() -> YAML:
    """Get a ruamel.yaml YAML handler with safe mode."""
    yaml = YAML(typ="safe", pure=True)
    return yaml


def get_yaml_dumper() -> YAML:
    """Get a ruamel.yaml YAML handler with representers for Enum and Path objects."""
    yaml = YAML(typ="rt", pure=True)

    # Use a very large width so long strings (e.g., file paths or keys with spaces)
    # are kept on a single line instead of being wrapped, which can otherwise cause
    # them to be emitted as complex keys. See also:
    # https://stackoverflow.com/questions/31197268/pyyaml-yaml-dump-produces-complex-key-for-string-key-122-chars/31199123#31199123
    # See PR https://github.com/DeepLabCut/DeepLabCut/pull/3140 for more details.
    yaml.width = 1_000_000

    # Auto-serialize Path objects as strings
    yaml.representer.add_multi_representer(PurePath, lambda r, p: r.represent_str(str(p)))
    yaml.representer.add_multi_representer(Enum, lambda r, e: r.represent_str(e.value))
    return yaml


def read_config_as_dict(config_path: str | Path) -> dict:
    """
    Args:
        config_path: the path to the configuration file to load

    Returns:
        The configuration file with pure Python classes

    Raises:
        FileNotFoundError: if the config file does not exist
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config {config_path} is not found. Please make sure that the file exists.")
    with open(config_path) as f:
        cfg = get_yaml_loader().load(f)
    if cfg is None:
        raise ValueError(f"Config {config_path} is empty or null.")
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {config_path} must be a YAML mapping at the top level, got {type(cfg).__name__}.")
    return cfg


def write_config(config_path: str | Path, config: dict, overwrite: bool = True) -> None:
    """Writes a pose configuration file to disk.

    Args:
        config_path: the path where the config should be saved
        config: the config to save
        overwrite: whether to overwrite the file if it already exists

    Raises:
        FileExistsError if overwrite=True and the file already exists
    """
    if not overwrite and Path(config_path).exists():
        raise FileExistsError(f"Cannot write to {config_path} - set overwrite=True to force")

    with open(config_path, "w") as file:
        get_yaml_dumper().dump(config, file)


def resolve_alias(
    name: str,
    alias_map: dict[str, str],
    *,
    warn: bool = True,
    stacklevel: int = 3,
) -> str:
    """Resolve a config key to its canonical field name.
    Args:
        name: Raw key name (alias or canonical).
        alias_map: ``{alias: canonical_name}`` for deprecated keys.
        warn: If True, emit :class:`DLCDeprecationWarning` when ``name`` is an alias.
        stacklevel: Passed to :func:`warnings.warn` for deprecation messages.
    Returns:
        Canonical field name, or ``name`` unchanged if it is not an alias.
    """
    canonical = alias_map.get(name, name)
    if warn and name in alias_map:
        from dlclibrary.deprecation import DLCDeprecationWarning

        warnings.warn(
            f"'{name}' is deprecated, use '{canonical}' instead.",
            DLCDeprecationWarning,
            stacklevel=stacklevel,
        )
    return canonical


def resolve_aliases_in_dict(
    cfg_dict: dict,
    alias_map: dict[str, str],
    *,
    target: str = "config",
    warn: bool = True,
    stacklevel: int = 3,
) -> dict:
    """Rename deprecated config keys to their canonical names.

    Args:
        cfg_dict: Raw configuration mapping (e.g. from YAML).
        alias_map: ``{alias: canonical_name}`` for deprecated keys.
        target: Config class name shown in errors.
        stacklevel: Passed to :func:`warnings.warn` for deprecation messages.

    Returns:
        A new dict with alias keys replaced by canonical names. Unchanged if
        ``alias_map`` is empty.

    Raises:
        TypeError: If multiple keys resolve to the same canonical field name
        (e.g. an alias and its canonical name, or two aliases for one field).
    """
    if not alias_map:
        return cfg_dict

    def _raise_for_duplicates(raw_to_canonical: dict[str, str]):
        counts = Counter(raw_to_canonical.values())
        conflicts = [f"{raw} -> {canonical}" for raw, canonical in raw_to_canonical.items() if counts[canonical] > 1]
        if conflicts:
            raise TypeError(f"{target} received duplicate canonical field names: {', '.join(conflicts)}.")

    raw_to_canonical = {raw: resolve_alias(raw, alias_map, warn=warn, stacklevel=stacklevel + 1) for raw in cfg_dict}
    _raise_for_duplicates(raw_to_canonical)
    return {raw_to_canonical[raw]: v for raw, v in cfg_dict.items()}


def normalize_for_serialization(obj: Any) -> Any:
    """Recursively normalize Paths to strings and Enums to values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return type(obj)({k: normalize_for_serialization(v) for k, v in obj.items()})
    if isinstance(obj, tuple):
        return tuple(normalize_for_serialization(v) for v in obj)
    if isinstance(obj, (list, set)):
        return [normalize_for_serialization(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def pretty_print(
    config: dict,
    indent: int = 0,
    print_fn: Callable[[str], None] | None = None,
) -> None:
    """Prints a model configuration in a pretty and readable way.

    Args:
        config: the config to print
        indent: the base indent on all keys
        print_fn: custom function to call (simply calls ``print`` if None)
    """
    if print_fn is None:
        print_fn = print

    for k, v in config.items():
        if isinstance(v, dict):
            print_fn(f"{indent * ' '}{k}:")
            pretty_print(v, indent + 2, print_fn=print_fn)
        else:
            print_fn(f"{indent * ' '}{k}: {v}")

