from typing import NoReturn


class UnavailableModule:
    """
    Fake module that can be imported but raises when used.
    """

    def __init__(self, module: str) -> None:
        self.__module = module

    def __getattr__(self, item: str) -> NoReturn:
        raise RuntimeError(f"{self.__module} cannot be imported.")


try:
    import numpy
except ImportError:
    numpy = UnavailableModule("numpy")

try:
    import pandas
except ImportError:
    pandas = UnavailableModule("pandas")
