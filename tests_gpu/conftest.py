# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Make the src/ layout (tesseract_jax_gpu, tesseract_jax_gpu_b) and the
# tests_gpu package importable when running these tests in-tree.
_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT / "src", _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")
