from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")
Factory = Callable[..., T]


class Registry(Generic[T]):
    """Maps a config string onto a factory, with a readable error on a typo."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Factory[T]] = {}

    def register(self, name: str) -> Callable[[Factory[T]], Factory[T]]:
        def wrap(factory: Factory[T]) -> Factory[T]:
            if name in self._factories:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._factories[name] = factory
            return factory

        return wrap

    def get(self, name: str) -> Factory[T]:
        try:
            return self._factories[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} '{name}'; available: {self.names()}"
            ) from None

    def create(self, name: str, **kwargs: object) -> T:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories
