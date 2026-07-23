"""Tests for root logger state preservation during paddleocr imports."""

import logging
import io


def _simulate_import_chain(restore_root_level, restore_root_handlers):
    """Simulate the save/restore logic used in paddleocr/__init__.py."""
    cur_handlers = list(logging.getLogger().handlers)
    if len(cur_handlers) != len(restore_root_handlers) or any(
        a is not b for a, b in zip(cur_handlers, restore_root_handlers)
    ):
        logging.getLogger().handlers[:] = restore_root_handlers
    logging.getLogger().setLevel(restore_root_level)


def test_root_logger_level_is_restored_after_simulated_imports():
    root_logger = logging.getLogger()
    orig_level = root_logger.level

    saved_level = root_logger.level
    saved_handlers = list(root_logger.handlers)

    root_logger.setLevel(logging.WARNING)

    _simulate_import_chain(saved_level, saved_handlers)

    assert root_logger.level == orig_level


def test_root_logger_handlers_are_restored_after_simulated_addition():
    root_logger = logging.getLogger()

    orig_len = len(root_logger.handlers)
    saved_level = root_logger.level
    saved_handlers = list(root_logger.handlers)

    extra = logging.StreamHandler(io.StringIO())
    root_logger.addHandler(extra)

    _simulate_import_chain(saved_level, saved_handlers)

    assert len(root_logger.handlers) == orig_len
    assert extra not in root_logger.handlers


def test_root_logger_handlers_are_restored_after_simulated_removal():
    root_logger = logging.getLogger()

    orig_len = len(root_logger.handlers)
    saved_level = root_logger.level
    saved_handlers = list(root_logger.handlers)

    root_logger.handlers[:] = []

    _simulate_import_chain(saved_level, saved_handlers)

    assert len(root_logger.handlers) == orig_len


def test_no_op_when_root_logger_state_unchanged():
    root_logger = logging.getLogger()

    saved_level = root_logger.level
    saved_handlers = list(root_logger.handlers)

    _simulate_import_chain(saved_level, saved_handlers)

    assert root_logger.level == saved_level
    assert list(root_logger.handlers) == saved_handlers
