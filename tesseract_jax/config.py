# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime configuration for Tesseract-JAX.

Values are read from the environment at import time and can be changed
afterwards, either for the rest of the process::

    tesseract_jax.config.update("check_static_outputs", False)

or for a block::

    with tesseract_jax.config.set(check_static_outputs=False):
        ...
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

__all__ = ["config"]


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{name} is set to {raw!r}, which is not a boolean. Use one of "
        f"1/0, true/false, yes/no, on/off."
    )


class Config:
    """The settings Tesseract-JAX reads at call time.

    Attributes:
        check_static_outputs: Whether ``apply`` compares the non-array outputs in
            its response against the values ``abstract_eval`` reported at trace
            time, and warns when they differ. Turning this off also skips the
            work behind the comparison, so the response is flattened without
            keypaths. Set from ``TESSERACT_JAX_CHECK_STATIC_OUTPUTS``, default
            on.
    """

    _defaults: ClassVar[dict[str, tuple[str, bool]]] = {
        "check_static_outputs": (
            "TESSERACT_JAX_CHECK_STATIC_OUTPUTS",
            True,
        ),
    }

    def __init__(self) -> None:
        for name, (env_var, default) in self._defaults.items():
            object.__setattr__(self, name, _bool_from_env(env_var, default))

    def __setattr__(self, name: str, value: Any) -> None:
        self.update(name, value)

    def update(self, name: str, value: Any) -> None:
        """Set one setting for the rest of the process."""
        if name not in self._defaults:
            raise AttributeError(
                f"Unknown setting {name!r}. Known settings: {sorted(self._defaults)}."
            )
        object.__setattr__(self, name, bool(value))

    @contextmanager
    def set(self, **settings: Any) -> Iterator[None]:
        """Set settings for the duration of a block, then put them back."""
        previous = {}
        for name in settings:
            if name not in self._defaults:
                raise AttributeError(
                    f"Unknown setting {name!r}. Known settings: "
                    f"{sorted(self._defaults)}."
                )
            previous[name] = getattr(self, name)
        try:
            for name, value in settings.items():
                self.update(name, value)
            yield
        finally:
            for name, value in previous.items():
                object.__setattr__(self, name, value)


config = Config()
