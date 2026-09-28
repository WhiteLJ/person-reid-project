"""Typed actions emitted by the Qt frontend."""

from enum import Enum, auto


class UIAction(Enum):
    SELECT_TARGET = auto()
    REMOVE_TARGET = auto()
    OPEN_GALLERY = auto()
    QUIT = auto()
    CLEAR_TARGETS = auto()
    PAUSE_TOGGLE = auto()
    SAVE_TARGET = auto()
    DISCARD_TARGET = auto()
