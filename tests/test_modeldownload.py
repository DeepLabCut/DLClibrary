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
import pytest


def test_catdownload(tmp_path_factory):
    # TODO: just download the lightweight stuff..
    import dlclibrary, os

    folder = tmp_path_factory.mktemp("cat")
    dlclibrary.download_huggingface_model("full_cat", str(folder))

    assert os.path.exists(folder / "pose_cfg.yaml")
    assert os.path.exists(folder / "snapshot-75000.meta")
