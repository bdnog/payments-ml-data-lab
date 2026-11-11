"""Code shared by the payments services (schemas, logging)."""

from payments_shared.logging import configure_logging
from payments_shared.schemas import HealthResponse

__all__ = ["HealthResponse", "configure_logging"]
