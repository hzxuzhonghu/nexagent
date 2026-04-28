"""NexAgent — next-generation agentic personal assistant."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nexagent")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
