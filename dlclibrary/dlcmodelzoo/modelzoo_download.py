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
    """ Only for PyTorch models.

    Returns:
        The detectors available for the dataset.
    """
    return list(_load_pytorch_dataset_models(dataset)["detectors"].keys())


def get_available_models(dataset: str) -> list[str]:
    """ Only for PyTorch models.

    Returns:
        The pose models available for the dataset.
    """
    return list(_load_pytorch_dataset_models(dataset)["pose_models"].keys())


def _handle_downloaded_file(
    file_path: str, target_dir: str, rename_mapping: dict | None = None
):
    """Handle the downloaded file from HuggingFace"""
    file_name = os.path.basename(file_path)
    try:
        with tarfile.open(file_path, mode="r:gz") as tar:
            for member in tar:
                if not member.isdir():
                    fname = Path(member.name).name
                    tar.makefile(member, os.path.join(target_dir, fname))
    except tarfile.ReadError:  # The model is a .pt file
        if rename_mapping is not None:
            file_name = rename_mapping.get(file_name, file_name)
        if os.path.islink(file_path):
            file_path_ = os.readlink(file_path)
            if not os.path.isabs(file_path_):
                file_path_ = os.path.abspath(
                    os.path.join(os.path.dirname(file_path), file_path_)
                )
            file_path = file_path_
        os.rename(file_path, os.path.join(target_dir, file_name))


def download_huggingface_model(
    model_name: str,
    target_dir: str = ".",
    remove_hf_folder: bool = True,
    rename_mapping: dict | None = None,
):
    """
    Downloads a DeepLabCut Model Zoo Project from Hugging Face.

    Args:
        model_name (str): Name of the ModelZoo model.
            For visualizations, see http://www.mackenziemathislab.org/dlc-modelzoo.
        target_dir (str): Directory where the model weights and pose_cfg.yaml file will be stored.
        remove_hf_folder (bool, optional): Whether to remove the directory structure provided by HuggingFace
            after downloading and decompressing the data into DeepLabCut format. Defaults to True.
        rename_mapping (dict, optional): A dictionary to rename the downloaded file.
            If None, the original filename is used. Defaults to None.
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

    for url in urls:
        url = url.split("/")
        repo_id, targzfn = url[0] + "/" + url[1], str(url[-1])

        hf_hub_download(repo_id, targzfn, cache_dir=str(target_dir))

        # Create a new subfolder as indicated below, unzipping from there and deleting this folder
        hf_folder = f"models--{url[0]}--{url[1]}"
        path_ = os.path.join(target_dir, hf_folder, "snapshots")
        commit = os.listdir(path_)[0]
        file_name = os.path.join(path_, commit, targzfn)
        _handle_downloaded_file(file_name, target_dir, rename_mapping)

    if remove_hf_folder:
        import shutil

        shutil.rmtree(os.path.join(target_dir, hf_folder))
