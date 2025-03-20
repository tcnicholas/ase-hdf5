from __future__ import annotations

import warnings
from pathlib import Path

import ase
import h5py
import numpy as np


class ASEH5Trajectory:
    def __init__(
        self,
        immutable: list[str] | None = None,
        mutable: list[str] | None = None,
        info_keys: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        immutable
            List of immutable properties to write to the HDF5 file.
        mutable
            List of mutable properties to write to the HDF5 file.
        info_keys
            List of per-frame info keys to write to the HDF5 file.
        """

        default_immutable = ["numbers"]
        default_mutable = ["positions"]

        # merge defaults with provided lists
        immutable = (immutable or []) + default_immutable
        mutable = (mutable or []) + default_mutable

        # Validate keys
        self.immutable, self.mutable = validate_keys(immutable, mutable)
        self.info_keys = info_keys or []

    def write(self, atoms_list: list[ase.Atoms], filename: Path | str) -> None:
        """
        Write a list of ASE Atoms objects to an HDF5 file.

        Some properties are considered immutable, and are stored in the
        "immutable" group. These are properties that should not change between
        frames, such as atomic numbers. Mutable properties are stored in the
        "mutable" group, and are properties that can change between frames, such
        as atomic positions. The cell is considered mutable if it changes
        between frames, and immutable otherwise.

        Basic checks are performed to ensure that immutable properties do not
        change between frames and that mutable properties are present in all
        frames.

        Parameters
        ----------
        atoms_list
            List of ASE Atoms objects to write.
        filename
            Path to the HDF5 file to write to.

        Raises
        ------
        ValueError
            If a mutable property is missing in any frame.
        """

        filename = Path(filename)

        with h5py.File(filename, "w") as h5file:
            first_atoms = atoms_list[0]

            # handle immutable properties.
            for key in self.immutable:
                data = first_atoms.arrays.get(key, first_atoms.info.get(key))
                if data is not None:
                    check_immutable_consistency(atoms_list, key, data)
                    h5file.create_dataset(f"immutable/{key}", data=data)
                else:
                    raise ValueError(
                        f"Immutable property '{key}' missing in frame 1."
                    )

            # handle mutable properties.
            for key in self.mutable:
                frame_data = []
                for atoms in atoms_list:
                    data = atoms.arrays.get(key, atoms.info.get(key))
                    if data is None:
                        raise ValueError(
                            f"Mutable property '{key}' missing in a frame."
                        )
                    frame_data.append(data)
                stacked_data = np.stack(frame_data, axis=0)
                h5file.create_dataset(f"mutable/{key}", data=stacked_data)

            # get the cell data.
            cells = np.stack([atoms.cell.array for atoms in atoms_list], axis=0)
            if np.all(np.isclose(cells, cells[0])):
                h5file.create_dataset("immutable/cell", data=cells[0])
            else:
                h5file.create_dataset("mutable/cell", data=cells)

            # other per-frame info.
            for key in self.info_keys:
                data = np.array(
                    [atoms.info.get(key, np.nan) for atoms in atoms_list]
                )
                if np.any(np.isnan(data)):
                    warnings.warn(
                        f"Some frames missing '{key}' info.", stacklevel=2
                    )
                h5file.create_dataset(f"info/{key}", data=data)

    def read(self, filename: Path | str) -> list[ase.Atoms]:
        """
        Read ASE Atoms objects from an HDF5 file.

        Parameters
        ----------
        filename
            Path to the HDF5 file to read from.

        Returns
        -------
        atoms_list
            List of ASE Atoms objects read from the file.
        """

        filename = Path(filename)

        atoms_list = []
        with h5py.File(filename, "r") as h5file:
            immutable_data = {
                key: np.array(val) for key, val in h5file["immutable"].items()
            }
            mutable_data = {
                key: np.array(val) for key, val in h5file["mutable"].items()
            }

            info_data = {}
            if "info" in h5file:
                info_data = {
                    key: np.array(val) for key, val in h5file["info"].items()
                }

            num_frames = next(iter(mutable_data.values())).shape[0]

            for i in range(num_frames):
                arrays = {
                    key: val[i]
                    for key, val in mutable_data.items()
                    if key != "cell"
                }
                arrays.update(immutable_data)

                cell = (
                    mutable_data.get("cell")[i]
                    if "cell" in mutable_data
                    else immutable_data.get("cell")
                )

                atoms = ase.Atoms(positions=arrays.pop("positions"), cell=cell)

                for key, val in arrays.items():
                    atoms.arrays[key] = val

                for key, val in info_data.items():
                    atoms.info[key] = val[i]

                atoms_list.append(atoms)

        return atoms_list

    def __repr__(self) -> str:
        """A string representation of the ASEH5Trajectory object."""

        indent = " " * 4

        def format_keys(keys_set):
            return (
                ",\n".join(f"{indent * 2}{key}" for key in keys_set)
                or f"{indent * 2}<none>"
            )

        mutable_keys = format_keys(self.mutable)
        immutable_keys = format_keys(self.immutable)
        info_keys = format_keys(self.info_keys)

        ensemble = "NPT" if "cell" in self.mutable else "NVT"

        repr_lines = [
            f"{indent}immutable_keys=(\n{immutable_keys}\n{indent})",
            f"{indent}mutable_keys=(\n{mutable_keys}\n{indent})",
        ]

        if self.info_keys:
            repr_lines.append(
                f"info_keys=(\n{info_keys}\n{indent})",
            )

        return (
            f"ASEH5Trajectory(\n{indent}ensemble={ensemble},\n"
            + ",\n".join(repr_lines)
            + "\n)"
        )


########## HELPER FUNCTIONS ##########


def validate_keys(
    immutable: list[str] | None, mutable: list[str] | None
) -> tuple[set[str], set[str]]:
    """
    Ensure there are no repeated keys in both immutable and mutable. If
    'numbers' appears in mutable, remove it from immutable. If 'positions'
    appears in immutable, remove it from mutable.

    Parameters
    ----------
    immutable
        List of immutable properties.
    mutable
        List of mutable properties.

    Returns
    -------
    Validated sets of immutable and mutable properties.
    """

    immutable_set = set(immutable or [])
    mutable_set = set(mutable or [])

    # Ensure 'numbers' and 'positions' are correctly placed.
    if "numbers" in mutable_set:
        immutable_set.discard("numbers")

    if "positions" in immutable_set:
        mutable_set.discard("positions")

    # Now remove common keys
    common_keys = immutable_set & mutable_set
    if common_keys:
        raise ValueError(
            "Conflicting keys found in both immutable and mutable: "
            + ", ".join([f"'{x}'" for x in common_keys])
        )

    return immutable_set, mutable_set


def check_immutable_consistency(atoms_list, key, data):
    """
    Check if an immutable property changes between frames.
    """
    for atoms in atoms_list[1:]:
        new_data = atoms.arrays.get(key, atoms.info.get(key))
        if new_data is not None and not np.array_equal(data, new_data):
            warnings.warn(
                f"Immutable property '{key}' changes between frames.",
                stacklevel=2,
            )
            break
