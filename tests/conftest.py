from __future__ import annotations

import sys
import types


if "ollama" not in sys.modules:
    ollama = types.ModuleType("ollama")

    class Client:  # pragma: no cover - test import shim
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    ollama.Client = Client
    sys.modules["ollama"] = ollama
