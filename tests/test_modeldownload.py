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


def test_catdownload():
    # TODO: just download the lightweight stuff..
    import dlclibrary, os

    os.mkdir("cat")
    dlclibrary.download_hugginface_model("full_cat", "cat")

    assert os.path.exists("cat/pose_cfg.yaml")
    assert os.path.exists("cat/snapshot-75000.meta")

    import shutil

    shutil.rmtree("cat")
