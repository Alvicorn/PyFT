"""
ShadowMap: maps (object_id, attr_name) -> VarState.
"""

from __future__ import annotations

import threading
import weakref

from ..core.var_state import VarState


class ShadowMap:
    """
    Registry: obj_id -> attr -> VarState.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # obj_id -> dict[attr, VarState]
        self._map: dict[int, dict[str, VarState]] = {}

        # obj_id -> weakref.finalize handle (kept alive)
        self._finalizers: dict[int, weakref.finalize] = {}

    def get_or_create(self, obj: object, attr: str) -> VarState:
        """
        Return the VarState for (obj, attr), creating it if needed.
        Also registers a GC finalizer on first access to this object.
        """
        oid = id(obj)
        with self._lock:
            if oid not in self._map:
                self._map[oid] = {}
                self._register_finalizer(obj, oid)
            attrs = self._map[oid]
            if attr not in attrs:
                attrs[attr] = VarState()
            return attrs[attr]

    def _register_finalizer(self, obj: object, oid: int) -> None:
        """
        Register a weakref finalizer to clean up when obj is GC'd.
        Caller holds self._lock.
        Some built-in types don't support weakrefs — we silently skip.
        """
        try:
            fin = weakref.finalize(obj, self._on_collect, oid)
            self._finalizers[oid] = fin
        except TypeError:
            pass  # weakref does not support int, str tuples, ...

    def _on_collect(self, oid: int) -> None:
        """Called by GC when the tracked object is collected."""
        with self._lock:
            self._map.pop(oid, None)
            self._finalizers.pop(oid, None)

    def remove_object(self, oid: int) -> None:
        """Explicitly remove all state for an object id."""
        with self._lock:
            self._map.pop(oid, None)
            fin = self._finalizers.pop(oid, None)
            if fin is not None:
                fin.detach()

    def stats(self) -> dict:
        with self._lock:
            n_objs = len(self._map)
            n_vars = sum(len(attrs) for attrs in self._map.values())
            return {"tracked_objects": n_objs, "tracked_variables": n_vars}
