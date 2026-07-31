"""Discovering what other packages have registered.

A trainer or a provider is a dotted path in a settings dict, which is fine when
you wrote it and tedious when someone else did: installing ``mlango-lightgbm``
should not also require knowing which module its class lives in and editing
``TRAINERS`` to say so.

So a package can advertise itself with a standard entry point:

    [project.entry-points."mlango.trainers"]
    lightgbm = "mlango_lightgbm.trainer:LightGBMTrainer"

and ``trainer = "lightgbm"`` works after ``pip install``. Nothing is imported
here — only names and paths are read, and resolution stays lazy, so a broken
plugin fails when something asks for it rather than at start-up.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

#: Setting name to the entry-point group that extends it.
GROUPS = {
    "TRAINERS": "mlango.trainers",
    "PROVIDERS": "mlango.providers",
}

logger = logging.getLogger("mlango.plugins")

_cache: dict[str, dict[str, str]] = {}


def discover(group: str) -> dict[str, str]:
    """``{name: dotted path}`` advertised by installed packages.

    Scanning distributions is not free and the answer cannot change while the
    process is running, so it is done once per group.
    """
    if group in _cache:
        return _cache[group]

    found: dict[str, str] = {}
    try:
        for entry in entry_points(group=group):
            # Entry points are written module:Object; mlango settings are
            # written module.Object. Accept the one and store the other, so a
            # plugin author writes normal packaging metadata and a user reading
            # `manage.py check` sees the form they would have typed themselves.
            found[entry.name] = entry.value.replace(":", ".", 1)
    except Exception:  # noqa: BLE001 - a broken distribution must not stop start-up
        logger.warning("Could not read %s entry points", group, exc_info=True)

    _cache[group] = found
    return found


def merged(setting: str, defaults: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    """Framework defaults, then installed plugins, then the project.

    The project wins on purpose: pointing ``TRAINERS["lightgbm"]`` at a patched
    subclass has to be possible without uninstalling the package that named it,
    and a plugin that could not be overridden would be a worse deal than the
    dotted path it replaced.
    """
    group = GROUPS.get(setting)
    out = dict(defaults)
    if group:
        out.update(discover(group))
    out.update(overrides)
    return out


def installed() -> dict[str, dict[str, str]]:
    """Everything discovered, keyed by setting — for ``manage.py check``."""
    return {setting: discover(group) for setting, group in GROUPS.items()}


def clear_cache() -> None:
    """Forget what was discovered. Used by tests."""
    _cache.clear()


__all__ = ["GROUPS", "discover", "merged", "installed", "clear_cache"]
