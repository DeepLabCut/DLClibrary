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
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlclibrary",
    version="0.0.2",
    author="A. & M. Mathis Labs",
    author_email="alexander@deeplabcut.org",
    description="Lightweight library supporting universal functions for the DeepLabCut ecosystem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DLClib",
    install_requires=[
        "huggingface_hub",
        "ruamel.yaml>=0.15.0",
    ],
    packages=setuptools.find_packages(),
    data_files=[
        (
            "dlclibrary",
            [
                "dlclibrary/modelzoo_urls.yaml",
            ],
        )
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
)
