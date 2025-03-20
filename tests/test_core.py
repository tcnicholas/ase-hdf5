import ase
import numpy as np
import pytest

from ase_hdf5.core import ASEH5Trajectory, validate_keys


@pytest.fixture(params=["immutable_cell", "mutable_cell"])
def sample_atoms_list(request):
    """
    Return a list of 10 random ASE Atoms objects, with either immutable (NVT) or
    mutable (NPT) cell.
    """

    cell = np.random.rand(3, 3)
    numbers = np.random.randint(1, 10, size=10)
    framework = np.random.randint(1, 10, size=10)
    mol_id = np.random.randint(1, 10, size=10)

    atoms_list = []
    for _ in range(10):
        atoms = ase.Atoms(
            numbers=numbers,
            positions=np.random.rand(10, 3),
            cell=(
                np.random.rand(3, 3)
                if request.param == "mutable_cell"
                else cell
            ),
        )
        atoms.arrays["framework"] = framework
        atoms.arrays["mol-id"] = mol_id
        atoms_list.append(atoms)

    return atoms_list


def test_write_and_read(tmp_path, sample_atoms_list):
    """Tests writing/reading ASE Atoms objects to and from an HDF5 file."""

    test_file = tmp_path / "test.h5"

    traj = ASEH5Trajectory(
        immutable=["numbers", "framework", "mol-id"],
        mutable=["positions"],
    )

    traj.write(sample_atoms_list, test_file)
    read_atoms_list = traj.read(test_file)

    assert len(read_atoms_list) == len(sample_atoms_list)
    for orig, read in zip(sample_atoms_list, read_atoms_list):
        np.testing.assert_array_equal(orig.numbers, read.numbers)
        np.testing.assert_array_equal(orig.positions, read.positions)
        np.testing.assert_array_equal(orig.cell.array, read.cell.array)


def test_immutable_property_warning(tmp_path, sample_atoms_list):
    """Tests warning when an immutable property changes between frames."""

    test_file = tmp_path / "test.h5"
    sample_atoms_list[5].arrays["framework"] = np.random.randint(1, 10, size=10)

    traj = ASEH5Trajectory(immutable=["framework"], mutable=["positions"])

    with pytest.warns(
        UserWarning,
        match="Immutable property 'framework' changes between frames.",
    ):
        traj.write(sample_atoms_list, test_file)


def test_missing_mutable_property_error(tmp_path, sample_atoms_list):
    """Tests error when a mutable property is missing in a frame."""

    test_file = tmp_path / "test.h5"
    del sample_atoms_list[5].arrays["positions"]

    traj = ASEH5Trajectory(immutable=["numbers"], mutable=["positions"])

    with pytest.raises(
        ValueError, match="Mutable property 'positions' missing in a frame."
    ):
        traj.write(sample_atoms_list, test_file)


def test_empty_atoms_list(tmp_path):
    """Tests writing an empty list of ASE Atoms objects."""

    test_file = tmp_path / "test.h5"
    traj = ASEH5Trajectory()

    with pytest.raises(IndexError):
        traj.write([], test_file)


def test_info_keys(tmp_path, sample_atoms_list):
    """Tests writing/reading info keys to and from an HDF5 file."""

    test_file = tmp_path / "test.h5"
    for i, atoms in enumerate(sample_atoms_list):
        atoms.info["energy"] = float(i)

    traj = ASEH5Trajectory(info_keys=["energy"])
    traj.write(sample_atoms_list, test_file)
    read_atoms_list = traj.read(test_file)

    assert all(
        atoms.info["energy"] == float(i)
        for i, atoms in enumerate(read_atoms_list)
    )


def test_info_keys_warning(tmp_path, sample_atoms_list):
    """Tests warning when an info key is missing in a frame."""

    test_file = tmp_path / "test.h5"
    for i, atoms in enumerate(sample_atoms_list):
        atoms.info["energy"] = float(i) if i % 2 == 0 else np.nan

    traj = ASEH5Trajectory(info_keys=["energy"])

    with pytest.warns(UserWarning, match="Some frames missing 'energy' info."):
        traj.write(sample_atoms_list, test_file)


def test_immutable_property_handling(tmp_path):
    """Tests behavior when immutable properties are missing in some frames."""

    test_file = tmp_path / "test.h5"

    # Create an immutable cell
    cell = np.random.rand(3, 3)
    numbers = np.random.randint(1, 10, size=10)

    # Generate atoms list
    atoms_list = []
    for i in range(10):
        atoms = ase.Atoms(
            numbers=numbers,
            positions=np.random.rand(10, 3),
            cell=cell,
        )

        # Ensure some frames are missing an immutable property
        if i % 2 == 0:
            atoms.arrays["framework"] = np.random.randint(1, 10, size=10)
        else:
            atoms.arrays["framework"] = np.random.randint(20, 30, size=10)

        atoms_list.append(atoms)

    # Create a trajectory object where "framework" is an immutable property
    traj = ASEH5Trajectory(
        immutable=["numbers", "framework"], mutable=["positions"]
    )

    with pytest.warns(
        UserWarning,
        match="Immutable property 'framework' changes between frames.",
    ):
        traj.write(atoms_list, test_file)

    # Read back the trajectory
    read_atoms_list = traj.read(test_file)

    # Verify that the read data contains numbers and positions
    assert len(read_atoms_list) == len(atoms_list)
    for orig, read in zip(atoms_list, read_atoms_list):
        np.testing.assert_array_equal(orig.numbers, read.numbers)
        np.testing.assert_array_equal(orig.positions, read.positions)

        # Check framework existence
        if "framework" in orig.arrays:
            assert "framework" in read.arrays
        else:
            assert "framework" not in read.arrays


def test_missing_immutable_property(tmp_path):
    """Tests behavior when an immutable property is missing in first frame."""

    test_file = tmp_path / "test.h5"

    # Create an immutable cell
    cell = np.random.rand(3, 3)
    numbers = np.random.randint(1, 10, size=10)
    framework = np.random.randint(1, 10, size=10)

    # Generate atoms list
    atoms_list = []
    for i in range(10):
        atoms = ase.Atoms(
            numbers=numbers,
            positions=np.random.rand(10, 3),
            cell=cell,
        )

        # Only set "framework" after the first frame (to trigger the ValueError)
        if i > 0:
            atoms.arrays["framework"] = framework

        atoms_list.append(atoms)

    # Create a trajectory object where "framework" is an immutable property
    traj = ASEH5Trajectory(
        immutable=["numbers", "framework"], mutable=["positions"]
    )

    # Expect a ValueError due to the missing immutable property in frame 0
    with pytest.raises(
        ValueError, match="Immutable property 'framework' missing in frame 1."
    ):
        traj.write(atoms_list, test_file)


def test_conflicting_keys():
    """
    Test if validate_keys raises an error for duplicate keys in immutable and
    mutable.
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting keys found in both immutable and mutable: 'shared_key'"
        ),
    ):
        validate_keys(["shared_key"], ["shared_key"])


def test_numbers_in_mutable():
    """Test if 'numbers' in mutable is removed from immutable."""

    immutable, mutable = validate_keys(["numbers"], ["numbers", "extra"])
    assert "numbers" not in immutable
    assert "extra" in mutable  # Ensure other mutable keys are still there


def test_positions_in_immutable():
    """Test if 'positions' in immutable is removed from mutable."""

    immutable, mutable = validate_keys(["positions", "extra"], ["positions"])
    assert "positions" not in mutable
    assert "extra" in immutable  # Ensure other immutable keys are still there
