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
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

import dlclibrary
import dlclibrary.dlcmodelzoo.modelzoo_download as modelzoo_download
from dlclibrary.dlcmodelzoo.modelzoo_download import MODELOPTIONS


def _fake_model_names():
    """
    Return a deterministic fake URL for each model.
    Alternate between tar.gz and .pt to test both branches.
    """
    mapping = {}
    for i, model in enumerate(MODELOPTIONS):
        ext = ".tar.gz" if i % 2 == 0 else ".pt"
        mapping[model] = f"fakeorg/{model}-repo/{model}{ext}"
    return mapping


def _write_fake_tar_gz(path: Path):
    """
    Create a fake tar.gz archive with the files the downloader expects
    for archive-based DLC models.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(path, mode="w:gz") as tar:
        files = {
            "pose_cfg.yaml": b"all_joints: [0, 1]\n",
            "snapshot-103000.index": b"fake index",
            "snapshot-103000.data-00000-of-00001": b"fake weights",
            "snapshot-103000.meta": b"fake meta",
        }

        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def _write_fake_pt(path: Path):
    """
    Create a fake .pt / .pth weight file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake pytorch weights")


@pytest.fixture
def mock_modelzoo(monkeypatch):
    """
    Patch both:
      - model name resolution
      - hf_hub_download network call

    so all downloads are local and deterministic.
    """
    fake_names = _fake_model_names()

    monkeypatch.setattr(modelzoo_download, "_load_model_names", lambda: fake_names)

    def fake_hf_hub_download(repo_id, filename, cache_dir):
        cache_dir = Path(cache_dir)
        hf_folder = cache_dir / f"models--{repo_id.replace('/', '--')}"
        snapshot_dir = hf_folder / "snapshots" / "fakecommit123"
        returned_file = snapshot_dir / filename

        if filename.endswith(".tar.gz"):
            _write_fake_tar_gz(returned_file)
        elif filename.endswith(".pt") or filename.endswith(".pth"):
            _write_fake_pt(returned_file)
        else:
            raise AssertionError(f"Unexpected mocked filename: {filename}")

        return str(returned_file)

    monkeypatch.setattr(modelzoo_download, "hf_hub_download", fake_hf_hub_download)

    return fake_names


def _assert_download_success(folder: Path, model: str):
    """
    Shared assertion helper for download_huggingface_model.
    """
    dlclibrary.download_huggingface_model(model, str(folder))

    files = {p.name for p in folder.iterdir()}

    # Archive-based DLC model
    if "pose_cfg.yaml" in files:
        assert "pose_cfg.yaml" in files
        assert any(name.startswith("snapshot-") for name in files)

    # Direct PyTorch model
    else:
        assert any(name.endswith((".pt", ".pth")) for name in files)

    # Verify that the Hugging Face cache folder was removed
    assert not any(name.startswith("models--") for name in files)


def test_download_huggingface_model_tar_or_pt(tmp_path, mock_modelzoo):
    folder = tmp_path / "download_one"
    folder.mkdir()

    # "full_cat" may map to tar.gz or .pt depending on ordering;
    # this assertion helper supports both branches.
    _assert_download_success(folder, "full_cat")


def test_download_huggingface_wrong_model(mock_modelzoo):
    with pytest.raises(ValueError):
        dlclibrary.download_huggingface_model("wrong_model_name")


def test_parse_superanimal_models():
    dict_ = dlclibrary.parse_available_supermodels()
    assert "superanimal_quadruped" in dict_
    assert "superanimal_topviewmouse" in dict_


@pytest.mark.parametrize("model", MODELOPTIONS)
def test_download_all_models(tmp_path, mock_modelzoo, model):
    folder = tmp_path / model
    folder.mkdir()
    _assert_download_success(folder, model)


def test_download_with_rename_mapping_for_pt(tmp_path, mock_modelzoo):
    """
    Explicitly test rename_mapping for a .pt model.
    """
    # Pick one of the mocked .pt models
    pt_model = None
    for i, model in enumerate(MODELOPTIONS):
        if i % 2 == 1:
            pt_model = model
            break

    assert pt_model is not None, "Expected at least one mocked .pt model"

    folder = tmp_path / "rename_pt"
    folder.mkdir()

    dlclibrary.download_huggingface_model(
        pt_model,
        str(folder),
        rename_mapping="renamed_weights.pt",
    )

    files = {p.name for p in folder.iterdir()}
    assert "renamed_weights.pt" in files
    assert not any(name.startswith("models--") for name in files)


def test_keep_hf_folder_when_requested(tmp_path, mock_modelzoo):
    """
    If remove_hf_folder=False, the cache structure should still exist.
    """
    folder = tmp_path / "keep_cache"
    folder.mkdir()

    dlclibrary.download_huggingface_model(
        "full_cat",
        str(folder),
        remove_hf_folder=False,
    )

    files = {p.name for p in folder.iterdir()}
    assert any(name.startswith("models--") for name in files)