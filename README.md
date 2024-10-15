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

PyTorch models available for a given dataset (compatible with DeepLabCut>=3.0) can be 
listed using the `dlclibrary.get_available_detectors` and 
`dlclibrary.get_available_models` methods. The datasets for which models are available
can be listed using `dlclibrary.get_available_datasets`. Example use:

```python
>>> import dlclibrary
>>> dlclibrary.get_available_datasets()
['superanimal_bird', 'superanimal_topviewmouse', 'superanimal_quadruped']

>>> dlclibrary.get_available_detectors("superanimal_bird")
['fasterrcnn_mobilenet_v3_large_fpn', 'ssdlite']

>>> dlclibrary.get_available_models("superanimal_bird")
['resnet_50']
```


## How to add a new model?

### TensorFlow models

Pick a good model_name. Follow the (novel) naming convention (modeltype_species), e.g. ```superanimal_topviewmouse```.  

1. Add the model_name with path and commit ID to: https://github.com/DeepLabCut/DLClibrary/blob/main/dlclibrary/dlcmodelzoo/modelzoo_urls.yaml

2. Add the model name to the constant: MODELOPTIONS
https://github.com/DeepLabCut/DLClibrary/blob/main/dlclibrary/dlcmodelzoo/modelzoo_download.py#L15

3. For superanimal models also fill in the configs!

### PyTorch models (for `deeplabcut >= 3.0.0`)

PyTorch models are listed in [`dlclibrary/dlcmodelzoo/modelzoo_urls_pytorch.yaml`](
https://github.com/DeepLabCut/DLClibrary/blob/main/dlclibrary/dlcmodelzoo/modelzoo_urls_pytorch.yaml
). The file is organized as:

```yaml
my_cool_dataset:  # name of the dataset used to train the model
  detectors:
    detector_name: path/to/huggingface-detector.pt  # add detectors under `detector`
  pose_models:
    pose_model_name: path/to/huggingface-pose-model.pt  # add pose models under `pose_models`
    other_pose_model_name: path/to/huggingface-other-pose-model.pt
```

This will allow users to download the models using the format `datatsetName_modelName`,
i.e. for this example 3 models would be available: `my_cool_dataset_detector_name`,
`my_cool_dataset_pose_model_name` and `my_cool_dataset_other_pose_model_name`.

To add a new model for `deeplabcut >= 3.0.0`, simply:

- add a new line under detectors or pose models if the dataset is already defined
- add the structure if the model was trained on a new dataset 

The models will then be listed when calling `dlclibrary.get_available_detectors` or
`dlclibrary.get_available_models`! You can list the datasets for which models are 
available using `dlclibrary.get_available_datasets`.
