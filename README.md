[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](README.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

# DLClibrary

DLClibrary is a lightweight library supporting universal functions for the [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) ecosystem.

Supported functions (at this point):

- API for downloading model weights from [the model zoo](http://www.mackenziemathislab.org/dlc-modelzoo)

# Quick start

## Install

The package can be installed using `pip`:

```bash
pip install dlclibrary
```

:warning: warning, the closely named package `dlclib` is not an official DeepLabCut product. :warning:

## Example Usage

Downloading a pretrained model from the model zoo:

```python
from pathlib import Path
from dlclibrary import download_huggingface_model

# Creates a folder and downloads the model to it
model_dir = Path("./superanimal_quadruped_model")
model_dir.mkdir()
download_huggingface_model("superanimal_quadruped", model_dir)
```

## How to add a new model?

Pick a good model_name. Follow the (novel) naming convention (modeltype_species), e.g. ```superanimal_topviewmouse```.  

1. Add the model_name with path and commit ID to: https://github.com/DeepLabCut/DLClibrary/blob/main/dlclibrary/dlcmodelzoo/modelzoo_urls.yaml

2. Add the model name to the constant: MODELOPTIONS
https://github.com/DeepLabCut/DLClibrary/blob/main/dlclibrary/dlcmodelzoo/modelzoo_download.py#L15

3. For superanimal models also fill in the configs!
