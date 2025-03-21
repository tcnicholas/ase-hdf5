# `ase-hdf5`

<div align="center">

![PyPI - Version](https://img.shields.io/pypi/v/ase-hdf5)
[![GitHub License](https://img.shields.io/github/license/tcnicholas/ase-hdf5)](LICENSE.md)
[![](https://github.com/tcnicholas/ase-hdf5/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/tcnicholas/ase-hdf5/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/tcnicholas/ase-hdf5/graph/badge.svg?token=LX45880JL6)](https://codecov.io/gh/tcnicholas/ase-hdf5)
</div>


`ase-hdf5` provides a simple I/O for converting list[ase.Atoms] <—> HDF5.


## Quickstart

Install the package with `pip install -q ase-hdf5`.

Then, a simple example of writing a list of ase.Atoms to a file:

```python
import ase
from ase_hdf5 import ASEH5Trajectory

atoms_list: list[ase.Atoms] = get_my_atoms_list() # with extra per-atom arrays.

traj_writer = ASEH5Trajectory(
    immutable=["numbers", "mol-id", "atom-type"],
    mutable=["positions"]
)

# write to file.
traj_writer.write(atoms_list, "atoms_list.h5")
````

We can run a simple check that the read-in version is equivalent:

```python
def atoms_are_equal(atoms1: ase.Atoms, atoms2: ase.Atoms) -> bool:
    """ Check if two ase.Atoms objects are equal. """

    _basic_properties = ["cell", "positions", "numbers"]
    _extra_properties = ["mol-id", "atom-type"]

    # all close because default writing converts to float32.
    for prop in _basic_properties:
        if not np.allclose(getattr(atoms1, prop), getattr(atoms2, prop)):
            return False
        
    for prop in _extra_properties:
        if (
            prop in atoms1.arrays 
            and prop in atoms2.arrays 
            and not np.array_equal(atoms1.arrays[prop], atoms2.arrays[prop])
        ):
            return False
        
    return True


# read in result.
atoms_list_read = traj_writer.read("atoms_list.h5")

for atom1, atom2 in zip(atoms_list, atoms_list_read):
    assert atoms_are_equal(atom1, atom2)
```