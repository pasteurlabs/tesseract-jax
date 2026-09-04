# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The settings object itself, without a Tesseract in the way."""

import pytest

from tesseract_jax import config
from tesseract_jax.config import Config, _bool_from_env


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("  On  ", True),
    ],
)
def test_the_environment_spells_booleans_several_ways(monkeypatch, raw, expected):
    monkeypatch.setenv("TESSERACT_JAX_TEST_FLAG", raw)
    assert _bool_from_env("TESSERACT_JAX_TEST_FLAG", not expected) is expected


def test_an_unset_variable_leaves_the_default(monkeypatch):
    monkeypatch.delenv("TESSERACT_JAX_TEST_FLAG", raising=False)
    assert _bool_from_env("TESSERACT_JAX_TEST_FLAG", True) is True
    assert _bool_from_env("TESSERACT_JAX_TEST_FLAG", False) is False


def test_a_value_that_is_not_a_boolean_says_so(monkeypatch):
    monkeypatch.setenv("TESSERACT_JAX_TEST_FLAG", "maybe")
    with pytest.raises(ValueError, match="not a boolean"):
        _bool_from_env("TESSERACT_JAX_TEST_FLAG", True)


def test_the_environment_sets_the_default_at_import(monkeypatch):
    monkeypatch.setenv("TESSERACT_JAX_CHECK_STATIC_OUTPUTS", "0")
    assert Config().check_static_outputs is False
    monkeypatch.setenv("TESSERACT_JAX_CHECK_STATIC_OUTPUTS", "1")
    assert Config().check_static_outputs is True


def test_checking_is_on_unless_asked_otherwise(monkeypatch):
    monkeypatch.delenv("TESSERACT_JAX_CHECK_STATIC_OUTPUTS", raising=False)
    assert Config().check_static_outputs is True


def test_a_setting_can_be_changed_by_assignment():
    assert config.check_static_outputs
    try:
        config.check_static_outputs = False
        assert config.check_static_outputs is False
    finally:
        config.check_static_outputs = True


def test_update_rejects_a_name_it_does_not_know():
    with pytest.raises(AttributeError, match="Unknown setting"):
        config.update("nonsense", True)


def test_the_block_form_rejects_a_name_it_does_not_know():
    with (
        pytest.raises(AttributeError, match="Unknown setting"),
        config.set(nonsense=True),
    ):
        pass
    assert config.check_static_outputs


def test_the_block_form_puts_the_value_back_after_a_failure():
    with pytest.raises(RuntimeError), config.set(check_static_outputs=False):
        raise RuntimeError("boom")
    assert config.check_static_outputs
