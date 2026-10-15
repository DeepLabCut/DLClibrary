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
"""Tests for dlclibrary.io.config."""

import warnings
from pathlib import Path

import pytest
from ruamel.yaml.error import YAMLError

from dlclibrary.io.config import (
    pretty_print,
    read_config_as_dict,
    resolve_alias,
    resolve_aliases_in_dict,
    write_config,
)
from dlclibrary.deprecation import DLCDeprecationWarning

# -----------------------------------------------------------------------------
# read_config_as_dict
# -----------------------------------------------------------------------------


def test_read_config_as_dict_loads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a: 1\nb: [2, 3]\nc:\n  d: 4\n")
    cfg = read_config_as_dict(str(config_path))
    assert cfg == {"a": 1, "b": [2, 3], "c": {"d": 4}}


def test_read_config_as_dict_raises_when_file_missing():
    with pytest.raises(FileNotFoundError):
        read_config_as_dict(Path("/nonexistent/config.yaml"))


def test_read_config_as_dict_raises_on_invalid_yaml(tmp_path):
    """Broken YAML syntax (e.g. unclosed bracket) raises an exception."""
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("key: [unclosed\n")
    with pytest.raises(YAMLError):
        read_config_as_dict(config_path)


@pytest.mark.parametrize("content", ["", "null", "~"])
def test_read_config_as_dict_raises_when_empty_or_null(tmp_path, content):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content)
    with pytest.raises(ValueError, match="empty or null"):
        read_config_as_dict(config_path)


def test_read_config_as_dict_raises_when_root_is_not_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- item\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        read_config_as_dict(config_path)


def test_read_config_as_dict_breaks_for_yaml_tags(tmp_path):
    """read_config breaks for YAML tags like !!python/tuple"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_path: /old/path\nengine: pytorch\nbodyparts: !!python/tuple [a, b, c]\n")
    with pytest.raises(YAMLError):
        read_config_as_dict(config_path)


def test_read_config_as_dict_accepts_config_with_misnamed_fields(tmp_path):
    """Config with typos/misnamed keys still loads; keys are not validated."""
    # NOTE @deruyter92 2026-02-03: This test captures the current behavior where
    # read_config_as_dict does not validate the keys. This behavior is different
    # from read_config, which should validate the keys (in the future).
    # The tests can be updated according, depending on the preferred behavior.
    config_path = tmp_path / "typos.yaml"
    config_path.write_text(
        "project_pathh: /wrong\n"  # typo
        "bodypartz: [a, b]\n"  # typo
        "Task: mytask\n"
    )
    cfg = read_config_as_dict(config_path)
    assert cfg["project_pathh"] == "/wrong"
    assert cfg["bodypartz"] == ["a", "b"]
    assert cfg["Task"] == "mytask"
    assert "project_path" not in cfg
    assert "bodyparts" not in cfg


# -----------------------------------------------------------------------------
# write_config
# -----------------------------------------------------------------------------


def test_write_config_creates_file(tmp_path):
    config_path = tmp_path / "out.yaml"
    write_config(config_path, {"a": 1, "b": 2})
    assert config_path.exists()
    cfg = read_config_as_dict(config_path)
    assert cfg["a"] == 1 and cfg["b"] == 2


def test_write_config_overwrites_existing(tmp_path):
    config_path = tmp_path / "out.yaml"
    config_path.write_text("old: true\n")
    write_config(config_path, {"new": True}, overwrite=True)
    cfg = read_config_as_dict(config_path)
    assert "new" in cfg and "old" not in cfg


def test_write_config_raises_when_overwrite_false_and_file_exists(tmp_path):
    config_path = tmp_path / "out.yaml"
    config_path.write_text("x: 1\n")
    with pytest.raises(FileExistsError):
        write_config(config_path, {"x": 2}, overwrite=False)


def test_write_config_allows_overwrite_false_when_file_missing(tmp_path):
    config_path = tmp_path / "new.yaml"
    write_config(config_path, {"x": 1}, overwrite=False)
    assert read_config_as_dict(config_path) == {"x": 1}


# -----------------------------------------------------------------------------
# pretty_print
# -----------------------------------------------------------------------------


def test_pretty_print_flat_config(capsys):
    pretty_print({"a": 1, "b": 2})
    out, _ = capsys.readouterr()
    assert "a: 1" in out and "b: 2" in out


def test_pretty_print_nested_config(capsys):
    pretty_print({"top": {"nested": 42}})
    out, _ = capsys.readouterr()
    assert "top:" in out and "nested: 42" in out


def test_pretty_print_with_indent(capsys):
    pretty_print({"k": "v"}, indent=4)
    out, _ = capsys.readouterr()
    assert out.startswith("    k:")


def test_pretty_print_with_custom_print_fn():
    lines = []

    def capture(s):
        lines.append(s)

    pretty_print({"a": 1}, print_fn=capture)
    assert any("a: 1" in line for line in lines)


# -----------------------------------------------------------------------------
# resolve_alias / resolve_aliases_in_dict
# -----------------------------------------------------------------------------


ALIAS_MAP = {"projectPath": "project_path"}


def test_resolve_alias_returns_canonical_for_alias():
    with pytest.warns(DLCDeprecationWarning, match="projectPath"):
        assert resolve_alias("projectPath", ALIAS_MAP) == "project_path"


def test_resolve_alias_returns_name_unchanged_for_unknown_key():
    assert resolve_alias("Task", ALIAS_MAP, warn=False) == "Task"


def test_resolve_alias_no_warning_when_warn_false():

    with warnings.catch_warnings():
        warnings.simplefilter("error", DLCDeprecationWarning)
        assert resolve_alias("projectPath", ALIAS_MAP, warn=False) == "project_path"


def test_resolve_aliases_in_dict_resolves_single_alias():
    with pytest.warns(DLCDeprecationWarning, match="projectPath"):
        resolved = resolve_aliases_in_dict(
            {"Task": "t", "projectPath": "/p"},
            ALIAS_MAP,
            target="ToyConfig",
        )
    assert resolved == {"Task": "t", "project_path": "/p"}


def test_resolve_aliases_in_dict_empty_alias_map_returns_input_unchanged():
    cfg_dict = {"Task": "t", "project_path": "/p"}
    assert resolve_aliases_in_dict(cfg_dict, {}) is cfg_dict


def test_resolve_aliases_in_dict_rejects_alias_and_canonical_together():
    with pytest.raises(TypeError, match=r"projectPath.*project_path"):
        resolve_aliases_in_dict(
            {"projectPath": "/alias", "project_path": "/canonical"},
            ALIAS_MAP,
            target="ToyConfig",
            warn=False,
        )


def test_resolve_aliases_in_dict_rejects_two_aliases_for_same_field():
    alias_map = {
        "projectPath": "project_path",
        "legacyProjectPath": "project_path",
    }
    with pytest.raises(TypeError, match=r"projectPath.*legacyProjectPath"):
        resolve_aliases_in_dict(
            {"projectPath": "/a", "legacyProjectPath": "/b"},
            alias_map,
            target="ToyConfig",
            warn=False,
        )
