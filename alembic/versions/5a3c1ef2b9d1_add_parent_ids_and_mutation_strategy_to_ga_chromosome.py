"""add parent ids and mutation strategy to GAChromosome

Revision ID: 5a3c1ef2b9d1
Revises: f0d711efe1a3
Create Date: 2025-06-17 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5a3c1ef2b9d1'
down_revision: Union[str, None] = 'f0d711efe1a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('ga_chromosomes', sa.Column('parent_ids', sa.JSON(), nullable=True))
    op.add_column('ga_chromosomes', sa.Column('mutation_strategy', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('ga_chromosomes', 'mutation_strategy')
    op.drop_column('ga_chromosomes', 'parent_ids')
