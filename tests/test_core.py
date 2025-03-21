import ase
import numpy as np
import pytest

from ase_hdf5.core import (
    ASEH5Trajectory,
    convert_dtype,
    decode_bytes,
    validate_keys,
)


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
        np.testing.assert_allclose(orig.positions, read.positions)
        np.testing.assert_allclose(orig.cell.array, read.cell.array)


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
        np.testing.assert_allclose(orig.positions, read.positions)

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


def test_read_no_cell_sets_no_pbc(tmp_path):
    """Ensure that if no cell is found, pbc is not set."""

    # create a valid file with all data *except* the cell.
    test_file = tmp_path / "no_cell.h5"
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0.8, 0, 0], [-0.4, 0.7, 0]])

    # write to file.
    traj = ASEH5Trajectory(immutable=["numbers"], mutable=["positions"])
    traj.write([atoms], test_file)

    # read back and test.
    atoms_list = traj.read(test_file)
    assert len(atoms_list) == 1
    assert not atoms_list[0].pbc.any()  # pbc should not be set.


########## TEST CONVERSIONS ##########


def test_convert_float_scalar():
    val = 3.14
    result = convert_dtype(val, np.float32)
    assert isinstance(result, np.float32)
    assert result == np.float32(3.14)


def test_convert_float_array():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = convert_dtype(arr, np.float32)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, arr, rtol=1e-6)


def test_convert_int_scalar():
    val = 42
    result = convert_dtype(val, np.float32)
    assert isinstance(result, int)  # Should remain unchanged
    assert result == 42


def test_convert_int_array():
    arr = np.array([1, 2, 3], dtype=np.int32)
    result = convert_dtype(arr, np.float32)
    assert result.dtype == np.int32  # Should remain unchanged
    np.testing.assert_array_equal(result, arr)


def test_convert_string_array():
    arr = np.array(["H", "C", "O"])
    result = convert_dtype(arr, np.float32)
    assert result.dtype.type is np.bytes_
    expected = np.array([b"H", b"C", b"O"])
    np.testing.assert_array_equal(result, expected)


def test_convert_object_array():
    arr = np.array([1, "H", 3.0], dtype=object)
    result = convert_dtype(arr, np.float32)
    assert result.dtype == object
    assert result.tolist() == arr.tolist()


def test_convert_empty_float_array():
    arr = np.array([], dtype=np.float64)
    result = convert_dtype(arr, np.float32)
    assert result.dtype == np.float32
    assert result.size == 0


def test_convert_empty_int_array():
    arr = np.array([], dtype=np.int64)
    result = convert_dtype(arr, np.float32)
    assert result.dtype == np.int64
    assert result.size == 0


def test_convert_already_correct_dtype():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = convert_dtype(arr, np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, arr)


def test_decode_byte_string_array():
    byte_array = np.array([b"H", b"O", b"C"], dtype="S1")
    result = decode_bytes(byte_array)
    
    assert result.dtype.kind == "U"  # Unicode
    assert result.dtype.itemsize >= 1
    assert result.tolist() == ["H", "O", "C"]


def test_unicode_string_array_unchanged():
    str_array = np.array(["H", "O", "C"], dtype="U1")
    result = decode_bytes(str_array)
    
    assert result is str_array  # should return original
    assert result.dtype.kind == "U"
    assert result.tolist() == ["H", "O", "C"]


def test_numeric_array_unchanged():
    float_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = decode_bytes(float_array)

    assert result is float_array
    np.testing.assert_array_equal(result, float_array)


def test_empty_byte_array():
    empty = np.array([], dtype="S1")
    result = decode_bytes(empty)

    assert result.dtype.kind == "U"
    assert result.size == 0


def test_non_ndarray_input():
    data = [b"H", b"O", b"C"]
    result = decode_bytes(data)

    assert result == data  # unchanged