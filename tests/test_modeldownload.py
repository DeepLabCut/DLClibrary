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
from dlclibrary.dlcmodelzoo.modelzoo_download import MODELOPTIONS


def test_download_huggingface_model(tmp_path_factory, model="full_cat"):
    folder = tmp_path_factory.mktemp("temp")
    dlclibrary.download_huggingface_model(model, str(folder))

    try:  # These are not created for .pt models
        assert os.path.exists(folder / "pose_cfg.yaml")
        assert any(f.startswith("snapshot-") for f in os.listdir(folder))
    except AssertionError:
        assert any(f.endswith(".pth") for f in os.listdir(folder))

    # Verify that the Hugging Face folder was removed
    assert not any(f.startswith("models--") for f in os.listdir(folder))


def test_download_huggingface_wrong_model():
    with pytest.raises(ValueError):
        dlclibrary.download_huggingface_model("wrong_model_name")


def test_parse_superanimal_models():
    dict_ = dlclibrary.parse_available_supermodels()
    assert "superanimal_quadruped" in dict_
    assert "superanimal_topviewmouse" in dict_


@pytest.mark.parametrize("model", MODELOPTIONS)
def test_download_all_models(tmp_path_factory, model):
    print("Downloading ...", model)
    test_download_huggingface_model(tmp_path_factory, model)
