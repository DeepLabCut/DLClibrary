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
"""Apply copyright headers to all code files in the repository.

This file can be called as a python script without arguments. For
configuration, see the instructions in NOTICE.yml.
"""

import fnmatch
import glob
import subprocess
import tempfile
from collections.abc import Iterable, Iterator

import yaml


def load_config(filename: str) -> list[dict]:
    """Load the list of header entries from a NOTICE.yml file."""
    with open(filename) as fh:
        return yaml.safe_load(fh)


def walk_directory(entry: dict) -> list[str]:
    """List the files an entry includes, minus those it excludes."""

    if "header" not in entry:
        raise ValueError("Current entry does not have a header.")
    if "include" not in entry:
        raise ValueError("Current entry does not have an include list.")

    def _list_include() -> Iterator[str]:
        """List all files specified in the include list."""
        for include_pattern in entry["include"]:
            yield from glob.iglob(include_pattern, recursive=True)

    def _filter_exclude(iterable: Iterable[str]) -> Iterator[str]:
        """Filter filenames from an iterator by the exclude patterns."""
        for filename in iterable:
            for exclude_pattern in entry.get("exclude", []):
                if fnmatch.fnmatch(filename, exclude_pattern):
                    break
            else:
                yield filename

    files = _filter_exclude(set(_list_include()))
    return list(files)


def main(input_file: str = "NOTICE.yml") -> None:
    """Write each entry's header into the files it covers, with licenseheaders."""
    config = load_config(input_file)
    for entry in config:
        filelist = list(walk_directory(entry))
        with tempfile.NamedTemporaryFile(mode="w") as header_file:
            header_file.write(entry["header"])
            header_file.flush()
            header_file.seek(0)
            command = ["licenseheaders", "-t", str(header_file.name), "-f", *filelist]
            result = subprocess.run(command, capture_output=True)
            if result.returncode != 0:
                print(result.stdout.decode())
                print(result.stderr.decode())


if __name__ == "__main__":
    main()
