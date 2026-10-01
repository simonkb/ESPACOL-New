"""Controlled benchmarks used by the ORIGIN acceptance-revision studies."""

from .ordinal_shortcut import (
    SHORTCUT_CONDITIONS,
    SHORTCUT_FAMILIES,
    FactorialShortcutDataset,
    OrdinalShortcutDataset,
    OrdinalShortcutProtocol,
    ShortcutMetadata,
    ShortcutSample,
    make_shortcut_loaders,
)

__all__ = [
    "SHORTCUT_CONDITIONS",
    "SHORTCUT_FAMILIES",
    "FactorialShortcutDataset",
    "OrdinalShortcutDataset",
    "OrdinalShortcutProtocol",
    "ShortcutMetadata",
    "ShortcutSample",
    "make_shortcut_loaders",
]
