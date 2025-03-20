from __future__ import annotations

from pathlib import Path


def human_readable_size(size_bytes: int) -> str:
    """ Convert bytes to a human-readable format (KB, MB, GB). """

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

    return f"{size_bytes:.2f} TB"


def get_file_size(file_path: Path) -> int:
    """ Get the size of a file in bytes. """

    return file_path.stat().st_size


def print_file_size(file_path: Path | str) -> None:
    """ Print the size of a file in a human-readable format. """

    size_bytes = get_file_size(Path(file_path))
    size_str = human_readable_size(size_bytes)
    print(f"File size: {size_str}")
