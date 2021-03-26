from typing import Any

_gradable = True


def gradable() -> bool:
    global _gradable

    return _gradable


class no_grad:
    def __init__(self) -> None:
        global _gradable

        self.prev = _gradable

    def __enter__(self) -> None:
        global _gradable

        self.prev = _gradable
        _gradable = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _gradable

        _gradable = self.prev
