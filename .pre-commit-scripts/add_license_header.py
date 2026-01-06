# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

HEADER_LINES = [
    "# This source code is licensed under the license found in the",
    "# LICENSE file in the root directory of this source tree.",
]
HEADER_BLOCK = "\n".join(HEADER_LINES) + "\n\n"


def is_header(lines):
    """Check if the lines exactly match the HEADER_LINES."""
    if len(lines) != len(HEADER_LINES):
        return False
    for line, expected in zip(lines, HEADER_LINES):
        if line.strip() != expected.strip():
            return False
    return True


def check_and_add_header(filepath):
    """Check if a file has a header; add it otherise. Return True if file was edited."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print(f"Skipping binary or non-utf8 file: {filepath}")
        return False

    header_line_index = 0
    if lines and lines[0].startswith("#!"):
        header_line_index = 1

    chunk_to_check = lines[header_line_index : header_line_index + len(HEADER_LINES)]
    if is_header(chunk_to_check):
        return False

    lines.insert(header_line_index, HEADER_BLOCK)

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Add license header to: {filepath}")
    return True


if __name__ == "__main__":
    files_changed = False

    for filename in sys.argv[1:]:
        if check_and_add_header(filename):
            files_changed = True

    if files_changed:
        sys.exit(1)
