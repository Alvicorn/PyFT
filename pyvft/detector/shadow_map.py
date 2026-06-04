from __future__ import annotations

import threading
import weakref
from typing import Callable, Union

from ..core.var_state import VarStateV1, VarStateV2

_real_lock = threading.Lock

_AnyVarState = Union[VarStateV1, VarStateV2]


class ShadowMap:
    """
    Registry of VarState objects keyed by ``(id(obj), attr)``.

    Registers a weakref finalizer on first access to each object so
    state is dropped when the object is garbage-collected. Types that
    do not support weakrefs (int, str, tuple, ...) are still tracked
    but their state leaks until ``clear`` is called.

    The ``var_state_factory`` parameter selects which VerifiedFT
    variant to instantiate (defaults to V2).
    """

    __slots__ = ("_lock", "_map", "_finalizers", "_factory")

    def __init__(
        self, var_state_factory: Callable[[], _AnyVarState] = VarStateV2
    ) -> None:
        self._lock = _real_lock()
        self._map: dict[int, dict[str, _AnyVarState]] = {}
        self._finalizers: dict[int, weakref.finalize] = {}
        self._factory = var_state_factory

    def get_or_create(self, obj: object, attr: str) -> _AnyVarState:
        oid = id(obj)
        with self._lock:
            if oid not in self._map:
                self._map[oid] = {}
                self._register_finalizer(obj, oid)
            attrs = self._map[oid]
            if attr not in attrs:
                attrs[attr] = self._factory()
            return attrs[attr]

    def _register_finalizer(self, obj: object, oid: int) -> None:
        try:
            fin = weakref.finalize(obj, self._on_collect, oid)
            self._finalizers[oid] = fin
        except TypeError:
            pass

    def _on_collect(self, oid: int) -> None:
        with self._lock:
            self._map.pop(oid, None)
            self._finalizers.pop(oid, None)

    def remove_object(self, oid: int) -> None:
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
