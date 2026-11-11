"""Create the runs table (the settlement work queue).

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(length=16), nullable=False),
        sa.Column("run_date", sa.Date(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("run_id"),
    )
    # One live run per date: a FAILED run is history and does not block a new request.
    op.create_index(
        "uq_runs_run_date_not_failed",
        "runs",
        ["run_date"],
        unique=True,
        postgresql_where=sa.text("status <> 'FAILED'"),
    )
    op.create_index("ix_runs_status_requested_at", "runs", ["status", "requested_at"])


def downgrade() -> None:
    op.drop_index("ix_runs_status_requested_at", table_name="runs")
    op.drop_index("uq_runs_run_date_not_failed", table_name="runs")
    op.drop_table("runs")
