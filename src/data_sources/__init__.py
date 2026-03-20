"""Data source connector registry.

Uses a registry pattern: each connector module calls @register("type_name")
to add itself to _REGISTRY when imported. The get_data_source() factory
looks up the type and instantiates the right connector class.
"""

from src.data_sources.base import BaseDataSource

# Maps type name strings (e.g., "local_pdf") to their connector classes.
# Populated lazily when _ensure_registered() imports connector modules.
_REGISTRY: dict[str, type[BaseDataSource]] = {}


def register(type_name: str):
    """Class decorator that registers a connector under the given type name."""

    def decorator(cls: type[BaseDataSource]):
        _REGISTRY[type_name] = cls
        return cls

    return decorator


def get_data_source(config: dict) -> BaseDataSource:
    """Factory function to instantiate a data source connector by type name.

    Args:
        config: Dict with at least a 'type' key and any source-specific fields.

    Returns:
        An instance of the appropriate BaseDataSource subclass.

    Raises:
        ValueError: If the type is unknown or not registered.
    """
    source_type = config.get("type")
    if not source_type:
        raise ValueError("Data source config missing 'type' field")

    # Lazy import connectors to populate registry
    _ensure_registered()

    cls = _REGISTRY.get(source_type)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown data source type '{source_type}'. "
            f"Available types: {available}"
        )

    return cls(config)


def _ensure_registered():
    """Lazily import connector modules so their @register decorators fire.

    Only runs once — subsequent calls short-circuit if _REGISTRY is populated.
    To add a new connector, add its import here.
    """
    if _REGISTRY:
        return

    # Each import triggers the @register decorator on the connector class
    from src.data_sources import local_pdf  # noqa: F401
