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
import pytest

import dlclibrary
import dlclibrary.dlcmodelzoo.modelzoo_download as modelzoo


@pytest.mark.parametrize(
    "data",
    [
        ("superanimal_bird", ["ssdlite"]),
        ("superanimal_topviewmouse", ["fasterrcnn_resnet50_fpn_v2"]),
        ("superanimal_quadruped", ["fasterrcnn_resnet50_fpn_v2"]),
    ]
)
def test_get_super_animal_detectors(data: tuple[str, list[str]]):
    dataset, expected_detectors = data
    detectors = modelzoo.get_available_detectors(dataset)
    assert len(detectors) >= len(expected_detectors)
    for det in expected_detectors:
        assert det in detectors


@pytest.mark.parametrize(
    "data",
    [
        ("superanimal_bird", ["resnet_50"]),
        ("superanimal_topviewmouse", ["hrnet_w32"]),
        ("superanimal_quadruped", ["hrnet_w32"]),
    ]
)
def test_get_super_animal_pose_models(data: tuple[str, list[str]]):
    dataset, expected_pose_models = data
    pose_models = modelzoo.get_available_models(dataset)
    assert len(pose_models) >= len(expected_pose_models)
    for pose_model in expected_pose_models:
        assert pose_model in pose_models
