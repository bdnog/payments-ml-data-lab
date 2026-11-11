"""Pure settlement and reconciliation rules.

Rule: no imports from Polars, FastAPI or storage in this package. Both the ``batch`` and ``api``
modules depend on ``domain``; ``domain`` depends on neither of them.
"""
