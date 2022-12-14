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
import dlclibrary
import os
import pytest


def test_download_huggingface_model(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cat")
    dlclibrary.download_huggingface_model("full_cat", str(folder))

    assert os.path.exists(folder / "pose_cfg.yaml")
    assert os.path.exists(folder / "snapshot-75000.meta")
    # Verify that the Hugging Face folder was removed
    assert not any(f.startswith("models--") for f in os.listdir(folder))


def test_download_huggingface_wrong_model():
    with pytest.raises(ValueError):
        dlclibrary.download_huggingface_model("wrong_model_name")
