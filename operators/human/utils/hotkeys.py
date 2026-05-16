"""Letter-key hotkey listener for the human operator.

Pynput delivers each key press to all active listeners, so this
coexists with the SO-101 leader's own listener (which owns the arrow
keys for slider drive). We only catch single printable letters in
``KEYS`` and surface them to the main loop on each tick.
"""
from __future__ import annotations

import threading

from pynput import keyboard as pynput_keyboard


class Hotkeys:
    KEYS = {"c", "x"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def pop_pressed(self) -> list[str]:
        with self._lock:
            keys = self._pending
            self._pending = []
        return keys

    def _on_press(self, key) -> None:
        char = getattr(key, "char", None)
        if char in self.KEYS:
            with self._lock:
                self._pending.append(char)
