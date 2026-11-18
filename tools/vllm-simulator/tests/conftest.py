# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def model_path():
    root = os.environ.get("SIMULATOR_TEST_MODEL_DIR")
    return lambda model: str(Path(root) / model) if root else model
