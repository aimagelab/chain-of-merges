from typing import Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str):
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' registered twice")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)


METHODS = Registry("method")
