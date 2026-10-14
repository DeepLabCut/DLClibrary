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

"""Check that every file NOTICE.yml covers starts with its header.

Reads only; use update_license_headers.py for writing.
"""

import sys
from itertools import takewhile
from pathlib import Path

from update_license_headers import load_config, walk_directory


def leading_comment(path: Path) -> list[str]:
    """Return the file's first comment block, with comment markers and one following space removed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.lstrip("#").removeprefix(" ").rstrip() for line in takewhile(lambda line: line.startswith("#"), lines)]


def has_header(path: Path, header: list[str]) -> bool:
    """Return whether the header lines appear, in order and contiguous, in the file's first comment block."""
    comment = leading_comment(path)
    return any(comment[start : start + len(header)] == header for start in range(len(comment) - len(header) + 1))


def main(input_file: str = "NOTICE.yml") -> int:
    """Print each file missing its header and return 1 if there is any, else 0."""
    missing = []
    for entry in load_config(input_file):
        header = [line.rstrip() for line in entry["header"].splitlines()]
        missing += [name for name in walk_directory(entry) if not has_header(Path(name), header)]

    for name in sorted(missing):
        print(f"{name}: missing or outdated license header (see {input_file})")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
