# The shared options and fixtures come from mblt_npu. Installed, they arrive
# through its pytest11 entry point and this file is unnecessary; from a source
# checkout the star import is what puts pytest_addoption in the root conftest
# namespace, which is the only place pytest looks for it.
from mblt_npu.pytest_plugin import *  # noqa: F401,F403
