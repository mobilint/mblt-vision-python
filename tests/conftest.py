# The shared options and fixtures come from mblt_npu. Installed, they arrive
# through its pytest11 entry point and this file is unnecessary; from a source
# checkout the star import is what puts pytest_addoption in the root conftest
# namespace, which is the only place pytest looks for it.
from pathlib import Path

import pytest
from PIL import Image

from mblt_npu.pytest_plugin import *  # noqa: F401,F403


@pytest.fixture
def synthetic_image_path(tmp_path: Path) -> Path:
    """Create a small RGB image for inference tests without committing assets."""

    image_path = tmp_path / "synthetic-image.jpg"
    Image.new("RGB", (64, 48), color=(64, 128, 192)).save(image_path)
    return image_path
