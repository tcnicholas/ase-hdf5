import pytest

from ase_hdf5.utils import _get_file_size, get_file_size, human_readable_size


@pytest.mark.parametrize(
    "size_bytes, expected_str, expected_float",
    [
        (500, "500.00 B", 500.00),
        (1023, "1023.00 B", 1023.00),
        (1024, "1.00 KB", 1.00),
        (1536, "1.50 KB", 1.50),
        (1048576, "1.00 MB", 1.00),
        (1073741824, "1.00 GB", 1.00),
        (1099511627776, "1.00 TB", 1.00),
    ],
)
def test_human_readable_size(size_bytes, expected_str, expected_float):
    """Test human_readable_size correctly converts bytes to readable format."""

    # Test default string output (auto unit selection)
    assert human_readable_size(size_bytes) == expected_str

    # Test float output
    assert human_readable_size(size_bytes, return_float=True) == pytest.approx(
        expected_float
    )


@pytest.mark.parametrize(
    "size_bytes, unit, expected_str, expected_float",
    [
        (500, "B", "500.00 B", 500.00),
        (500, "KB", "0.49 KB", 0.48828125),
        (1048576, "KB", "1024.00 KB", 1024.00),
        (1048576, "MB", "1.00 MB", 1.00),
        (1048576, "GB", "0.00 GB", 0.0009765625),
        (1073741824, "GB", "1.00 GB", 1.00),
        (1099511627776, "TB", "1.00 TB", 1.00),
    ],
)
def test_human_readable_size_specific_units(
    size_bytes, unit, expected_str, expected_float
):
    """Test human_readable_size conversion to a specific unit."""

    # Test string output for the requested unit
    assert human_readable_size(size_bytes, units=unit) == expected_str

    # Test float output for the requested unit
    assert human_readable_size(
        size_bytes, units=unit, return_float=True
    ) == pytest.approx(expected_float)


def test_get_file_size(tmp_path):
    """Test get_file_size by creating a temporary file and checking its size."""

    test_file = tmp_path / "test.txt"

    # Write 100 bytes to the file
    test_content = "a" * 100
    test_file.write_text(test_content)

    assert _get_file_size(test_file) == 100


def test_print_file_size(tmp_path):
    """Test print_file_size function captures correct output."""

    test_file = tmp_path / "test.txt"

    # Write 2048 bytes (2 KB).
    test_file.write_text("a" * 2048)
    file_size = get_file_size(test_file)

    assert "2.00 KB" in file_size
