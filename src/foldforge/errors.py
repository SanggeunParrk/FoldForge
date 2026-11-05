"""Public exceptions that callers can catch without importing inference backends."""


class FoldForgeError(Exception):
    """Base class for FoldForge lifecycle and input errors."""


class InputError(FoldForgeError, ValueError):
    """An input or per-call sampling option is invalid."""


class ModelClosedError(FoldForgeError, RuntimeError):
    """A closed predictor cannot execute inference."""


class ConcurrentCallError(FoldForgeError, RuntimeError):
    """A predictor or trunk state is already serving another operation."""


class TrunkStateError(FoldForgeError, ValueError):
    """A trunk state is closed or belongs to another predictor."""
