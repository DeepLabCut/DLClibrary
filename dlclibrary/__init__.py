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
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dlclibrary")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

VERSION = __version__
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    download_huggingface_model,
    get_available_datasets,
    get_available_detectors,
    get_available_models,
    parse_available_supermodels,
)
