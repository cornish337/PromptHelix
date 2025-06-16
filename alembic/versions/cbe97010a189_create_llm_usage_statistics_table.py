"""create_llm_usage_statistics_table

Revision ID: cbe97010a189
Revises: 61ac22f03a39
Create Date: 2025-06-16 14:34:22.227544

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cbe97010a189'
down_revision: Union[str, None] = '61ac22f03a39'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('llm_usage_statistics',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('llm_service', sa.String(), nullable=False),
        sa.Column('request_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.UniqueConstraint('llm_service', name='uq_llm_usage_service')
    )
    op.create_index(op.f('ix_llm_usage_statistics_llm_service'), 'llm_usage_statistics', ['llm_service'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_llm_usage_statistics_llm_service'), table_name='llm_usage_statistics')
    op.drop_table('llm_usage_statistics')
