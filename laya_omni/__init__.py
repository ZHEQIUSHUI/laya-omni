"""laya-omni: Laya typed decisions with optional image and audio inputs."""

__all__ = ["Agent", "Omni", "load"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy, so the game renderers and data tools work without torch installed.
    if name == "Omni":
        from .pipeline import Omni

        return Omni
    if name in __all__:
        from . import agent

        return getattr(agent, name)
    raise AttributeError(name)
