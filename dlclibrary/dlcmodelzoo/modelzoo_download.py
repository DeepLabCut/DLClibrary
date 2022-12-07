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
import os

# just expand this list when adding new models:
MODELOPTIONS = [
    "full_human",
    "full_cat",
    "primate_face",
    "mouse_pupil_vclose",
    "horse_sideview",
    "full_macaque",
    "superanimal_mouse",
]


def _get_dlclibrary_path():
    """Get path of where dlclibrary (this repo) is currently running"""
    import importlib.util

    return os.path.split(importlib.util.find_spec("dlclibrary").origin)[0]


def _loadmodelnames():
    """Loads URLs and commit hashes for available models."""
    from ruamel.yaml import YAML

    fn = os.path.join(_get_dlclibrary_path(), "modelzoo_urls.yaml")
    with open(fn) as file:
        return YAML().load(file)


def download_huggingface_model(modelname, target_dir=".", removeHFfolder=True):
    """
    Downloads a DeepLabCut Model Zoo Project from Hugging Face

    Parameters
    ----------
    modelname : string
        Name of the ModelZoo model. For visualizations see: http://www.mackenziemathislab.org/dlc-modelzoo
    target_dir : directory (as string)
        Directory where to store the model weigths and pose_cfg.yaml file
    removeHFfolder : bool, default True
        Whether to remove the directory structure provided by HuggingFace after downloading and decompressing data into DeepLabCut format.
    """
    from huggingface_hub import hf_hub_download
    import tarfile, os
    from pathlib import Path

    neturls = _loadmodelnames()

    if modelname in neturls.keys():
        print("Loading....", modelname)
        url = neturls[modelname].split("/")
        repo_id, targzfn = url[0] + "/" + url[1], str(url[-1])

        hf_hub_download(repo_id, targzfn, cache_dir=str(target_dir))
        # creates a new subfolder as indicated below, unzipping from there and deleting this folder

        # Building the HuggingFaceHub download path:
        hf_path = (
            "models--"
            + url[0]
            + "--"
            + url[1]
            + "/snapshots/"
            + str(neturls[modelname + "_commit"])
            + "/"
            + targzfn
        )

        filename = os.path.join(target_dir, hf_path)
        with tarfile.open(filename, mode="r:gz") as tar:
            for member in tar:
                if not member.isdir():
                    fname = Path(member.name).name  # getting the filename
                    tar.makefile(member, target_dir + "/" + fname)
                    # tar.extractall(target_dir, members=tarfilenamecutting(tar))

        if removeHFfolder:
            # Removing folder
            import shutil

            shutil.rmtree(
                Path(os.path.join(target_dir, "models--" + url[0] + "--" + url[1]))
            )

    else:
        models = [fn for fn in neturls.keys()]
        print("Model does not exist: ", modelname)
        print("Pick one of the following: ", MODELOPTIONS)


if __name__ == "__main__":
    print("Randomly downloading a model for testing...")

    import random

    # modelname = 'full_cat'
    modelname = random.choice(MODELOPTIONS)

    target_dir = "/Users/alex/Downloads"  # folder has to exist!
    download_hugginface_model(modelname, target_dir)
