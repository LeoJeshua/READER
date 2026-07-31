from pathlib import Path

import pytest

from tools.validate_release import validate_release_boundary


def test_boundary_ignores_local_workspace(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "features").symlink_to(tmp_path / "external-cache")
    private = tmp_path / "private"
    private.mkdir()
    (private / "endpoint.txt").write_text(
        "api" + "_key=" + "x" * 24, encoding="utf-8"
    )

    assert validate_release_boundary(tmp_path) == {
        "text_files_scanned": 0,
        "symlinks": 0,
    }


def test_boundary_rejects_public_symlink(tmp_path: Path) -> None:
    (tmp_path / "published-link").symlink_to(tmp_path / "missing")

    with pytest.raises(ValueError, match="contains symlinks"):
        validate_release_boundary(tmp_path)


def test_boundary_rejects_literal_public_credential(tmp_path: Path) -> None:
    (tmp_path / "settings.txt").write_text(
        "api" + "_key=" + "x" * 24, encoding="utf-8"
    )

    with pytest.raises(ValueError, match="boundary violations"):
        validate_release_boundary(tmp_path)
