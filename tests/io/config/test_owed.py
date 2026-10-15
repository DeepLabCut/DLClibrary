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
"""Owed changes relative to DeepLabCut's copy of the config code."""

from pathlib import Path

import pytest

from dlclibrary.io.config import ProjectConfig


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="ProjectConfig._post_yaml_load_updates replaces project_path on every from_yaml, as in DeepLabCut",
)
def test_a_plain_read_leaves_project_path_as_stored(tmp_path):
    stored = tmp_path / "elsewhere"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"project_path: {stored.as_posix()}\n")

    assert ProjectConfig.from_yaml(config_path).project_path == Path(stored.as_posix())
