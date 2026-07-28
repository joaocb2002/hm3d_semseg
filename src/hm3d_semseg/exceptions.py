"""Project-specific exceptions with actionable failure messages."""


class HM3DSemsegError(RuntimeError):
    """Base exception for expected command failures."""


class ConfigurationError(HM3DSemsegError):
    """Raised when configuration is missing, unknown, or inconsistent."""


class CameraContractError(HM3DSemsegError):
    """Raised when camera geometry cannot be resolved or does not match."""


class DatasetValidationError(HM3DSemsegError):
    """Raised when an offline dataset violates its schema or invariants."""


class OptionalDependencyError(HM3DSemsegError):
    """Raised when an explicitly requested optional workflow lacks a dependency."""
