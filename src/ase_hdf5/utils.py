from __future__ import annotations

from pathlib import Path


def human_readable_size(
    size_bytes: int, units: str | None = None, return_float: bool = False
) -> str | float:
    """
    Convert bytes to a human-readable format (B, KB, MB, GB, TB).

    Parameters
    ----------
    size_bytes : int
        The size in bytes to convert.
    units : str, optional
        The unit to return the size in (B, KB, MB, GB, TB).
    return_float : bool, optional
        If True, return the size as a float without units.

    Returns
    -------
    str | float
        The size converted to the requested or most appropriate unit.
    """

    unit_list = ["B", "KB", "MB", "GB"]
    size_bytes = float(size_bytes)  # Ensure float precision

    # If a specific unit is requested, convert directly to that unit
    if units and units in unit_list:
        unit_index = unit_list.index(units)
        size_bytes /= 1024**unit_index
        return size_bytes if return_float else f"{size_bytes:.2f} {units}"

    # Otherwise, find the appropriate unit dynamically
    for unit in unit_list:
        if size_bytes < 1024:
            return size_bytes if return_float else f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

    # If it's beyond TB, return TB representation
    return size_bytes if return_float else f"{size_bytes:.2f} TB"


def _get_file_size(file_path: Path) -> int:
    """Get the size of a file in bytes."""

    return file_path.stat().st_size


def get_file_size(file_path: Path | str) -> str:
    """Get the size of a file in a human-readable format."""

    size_bytes = _get_file_size(Path(file_path))
    return human_readable_size(size_bytes)
