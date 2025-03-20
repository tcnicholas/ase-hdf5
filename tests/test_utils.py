import pytest

from ase_hdf5.utils import get_file_size, human_readable_size, print_file_size


@pytest.mark.parametrize(
    "size_bytes, expected",
    [
        (500, "500.00 B"),
        (1023, "1023.00 B"),
        (1024, "1.00 KB"),
        (1536, "1.50 KB"),
        (1048576, "1.00 MB"),
        (1073741824, "1.00 GB"),
        (1099511627776, "1.00 TB"),
    ],
)
def test_human_readable_size(size_bytes, expected):
    """Test human_readable_size correctly converts bytes to readable format."""

    assert human_readable_size(size_bytes) == expected


def test_get_file_size(tmp_path):
    """Test get_file_size by creating a temporary file and checking its size."""

    test_file = tmp_path / "test.txt"

    # Write 100 bytes to the file
    test_content = "a" * 100
    test_file.write_text(test_content)

    assert get_file_size(test_file) == 100


def test_print_file_size(tmp_path, capsys):
    """Test print_file_size function captures correct output."""

    test_file = tmp_path / "test.txt"

    # Write 2048 bytes (2 KB)
    test_file.write_text("a" * 2048)

    print_file_size(test_file)

    captured = capsys.readouterr()
    assert "File size: 2.00 KB" in captured.out
