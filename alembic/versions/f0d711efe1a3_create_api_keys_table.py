"""create api_keys table

Revision ID: f0d711efe1a3
Revises: 334ec73d5186
Create Date: 2025-06-16 03:03:17.730574

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f0d711efe1a3'
down_revision: Union[str, None] = '334ec73d5186'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create api_keys table."""
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("service_name", sa.String(), nullable=False, unique=True),
        sa.Column("api_key", sa.String(), nullable=False),
        sa.UniqueConstraint("service_name", name="uq_service_name"),
    )
    op.create_index(op.f("ix_api_keys_id"), "api_keys", ["id"], unique=False)
    op.create_index(op.f("ix_api_keys_service_name"), "api_keys", ["service_name"], unique=True)


def downgrade() -> None:
    """Drop api_keys table."""
    op.drop_index(op.f("ix_api_keys_service_name"), table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_id"), table_name="api_keys")
    op.drop_table("api_keys")
