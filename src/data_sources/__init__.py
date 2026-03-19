"""Data source connectors and registry."""

from src.data_sources.base import BaseDataSource

# Registry mapping type strings to connector classes.
# Each connector module registers itself when imported.
_REGISTRY: dict[str, type[BaseDataSource]] = {}


def register(type_name: str):
    """Decorator to register a data source connector class."""

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


_ALL_CONNECTOR_MODULES = ("local_pdf", "local_txt", "local_csv", "gdrive", "s3", "notion", "web", "huggingface")


def _ensure_registered():
    """Import all connector modules to trigger registration."""
    if len(_REGISTRY) >= len(_ALL_CONNECTOR_MODULES):
        return

    # Import each connector module — their @register decorators populate _REGISTRY
    from src.data_sources import (  # noqa: F401
        local_pdf,
        local_txt,
        local_csv,
        gdrive,
        s3,
        notion,
        web,
        huggingface,
    )
