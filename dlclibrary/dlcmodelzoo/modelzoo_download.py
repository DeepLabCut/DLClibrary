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

import json
import os
import tarfile
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedBase

# just expand this list when adding new models:
MODELOPTIONS = [
    "full_human",
    "full_cat",
    "full_dog",
    "primate_face",
    "mouse_pupil_vclose",
    "horse_sideview",
    "full_macaque",
    "superanimal_quadruped",
    "superanimal_topviewmouse",
]


def _get_dlclibrary_path():
    """Get path of where dlclibrary (this repo) is currently running"""
    import importlib.util

    return os.path.split(importlib.util.find_spec("dlclibrary").origin)[0]


def _load_pytorch_models() -> dict[str, dict[str, dict[str, str]]]:
    """Load URLs and commit hashes for available models."""
    urls = Path(_get_dlclibrary_path()) / "dlcmodelzoo" / "modelzoo_urls_pytorch.yaml"
    with open(urls) as file:
        data = YAML(pure=True).load(file)

    return data


def _load_pytorch_dataset_models(dataset: str) -> dict[str, dict[str, str]]:
    """Load URLs and commit hashes for available models."""
    models = _load_pytorch_models()
    if not dataset in models:
        raise ValueError(
            f"Could not find any models for {dataset}. Models are available for "
            f"{list(models.keys())}"
        )

    return models[dataset]


def _load_model_names():
    """Load URLs and commit hashes for available models."""
    fn = os.path.join(_get_dlclibrary_path(), "dlcmodelzoo", "modelzoo_urls.yaml")
    with open(fn) as file:
        model_names = YAML().load(file)

    # add PyTorch models
    for dataset, model_types in _load_pytorch_models().items():
        for model_type, models in model_types.items():
            for model, url in models.items():
                model_names[f"{dataset}_{model}"] = url

    return model_names


def parse_available_supermodels():
    libpath = _get_dlclibrary_path()
    json_path = os.path.join(libpath, "dlcmodelzoo", "superanimal_models.json")
    with open(json_path) as file:
        super_animal_models = json.load(file)
    return super_animal_models


def get_available_datasets() -> list[str]:
    """Only for PyTorch models.

    Returns:
        The name of datasets for which models are available
    """
    return list(_load_pytorch_models().keys())


def get_available_detectors(dataset: str) -> list[str]:
    """Only for PyTorch models.

    Returns:
        The detectors available for the dataset.
    """
    return list(_load_pytorch_dataset_models(dataset)["detectors"].keys())


def get_available_models(dataset: str) -> list[str]:
    """Only for PyTorch models.

    Returns:
        The pose models available for the dataset.
    """
    return list(_load_pytorch_dataset_models(dataset)["pose_models"].keys())



def _handle_downloaded_file(
    file_path: str, target_dir: str, rename_mapping: dict | None = None
):
    """Handle the downloaded file from HuggingFace cache and place the final artifact in target_dir."""
    file_name = os.path.basename(file_path)

    try:
        # Be permissive about compression type
        with tarfile.open(file_path, mode="r:*") as tar:
            extracted_any = False
            for member in tar.getmembers():
                # Only extract regular files
                if not member.isfile():
                    continue

                fname = Path(member.name).name
                if not fname:
                    continue

                src = tar.extractfile(member)
                if src is None:
                    continue

                extracted_path = os.path.join(target_dir, fname)
                with src, open(extracted_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                extracted_any = True

            # If it opened as a tar but contained nothing useful, fail loudly
            if not extracted_any:
                raise tarfile.ReadError(f"No regular files extracted from archive: {file_path}")

    except tarfile.ReadError:
        # Not an archive -> treat as a direct model file (.pt/.pth/etc.)
        if rename_mapping is not None:
            file_name = rename_mapping.get(file_name, file_name)
        shutil.copy2(file_path, os.path.join(target_dir, file_name))


def download_huggingface_model(
    model_name: str,
    target_dir: str = ".",
    rename_mapping: str | dict | None = None,
):
    """
    Downloads a DeepLabCut Model Zoo Project from Hugging Face.

    Args:
        model_name (str):
            Name of the ModelZoo model.
            For visualizations, see http://www.mackenziemathislab.org/dlc-modelzoo.
        target_dir (str, optional):
            Target directory where the model weights will be stored.
            Defaults to the current directory.
        rename_mapping (dict | str | None, optional):
            - If a dictionary, it should map the original Hugging Face filenames
              to new filenames (e.g. {"snapshot-12345.tar.gz": "mymodel.tar.gz"}).
            - If a string, it is interpreted as the new name for the downloaded file
            - If None, the original filename is used.
            Defaults to None.

    Examples:
        >>> # Download without renaming, keep original filename
        download_huggingface_model("superanimal_bird_resnet_50")

        >>> # Download and rename by specifying the new name directly
        download_huggingface_model(
            model_name="superanimal_humanbody_rtmpose_x",
            target_dir="/path/to/,y/checkpoints",
            rename_mapping="superanimal_humanbody_rtmpose_x.pt"
        )
    """
    net_urls = _load_model_names()
    if model_name not in net_urls:
        raise ValueError(
            f"`modelname={model_name}` should be one of: {', '.join(net_urls)}."
        )

    print("Loading....", model_name)
    urls = net_urls[model_name]
    if isinstance(urls, CommentedBase):
        urls = list(urls)
    else:
        urls = [urls]

    if not os.path.isabs(target_dir):
        target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dlc_hf_") as hf_cache_dir:
        for url in urls:
            url = url.split("/")
            repo_id, targzfn = url[0] + "/" + url[1], str(url[-1])

            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=targzfn,
                cache_dir=hf_cache_dir,
            )

            if isinstance(rename_mapping, str):
                mapping = {targzfn: rename_mapping}
            else:
                mapping = rename_mapping

            _handle_downloaded_file(downloaded, target_dir, mapping)
