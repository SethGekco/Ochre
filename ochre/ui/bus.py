# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""A tiny Qt-free publish/subscribe bus.

The engine and the controller emit on this; Qt widgets subscribe. That
indirection is what lets the controller notify the UI without importing Qt,
and what lets the whole application be driven from a script with no widgets
subscribed at all.

A handler that raises is logged and removed rather than being allowed to
break the emitter -- one bad panel should not take down the editor.
"""


class EventBus:
    def __init__(self):
        self._subs = {}
        self.errors = []

    def subscribe(self, topic, handler):
        self._subs.setdefault(topic, []).append(handler)
        return handler

    def unsubscribe(self, topic, handler):
        handlers = self._subs.get(topic)
        if not handlers:
            return False
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    def emit(self, topic, **payload):
        delivered = 0
        for handler in list(self._subs.get(topic, ())):
            try:
                handler(**payload)
                delivered += 1
            except Exception as exc:                       # noqa: BLE001
                self.errors.append((topic, handler, exc))
                self.unsubscribe(topic, handler)
        return delivered

    def topics(self):
        return sorted(t for t, h in self._subs.items() if h)

    def clear(self):
        self._subs.clear()
        self.errors.clear()
