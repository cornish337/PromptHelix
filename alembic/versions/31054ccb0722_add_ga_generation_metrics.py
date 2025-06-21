"""add ga generation metrics

Revision ID: 31054ccb0722
Revises: 3ef25a559d5b
Create Date: 2025-06-21 04:17:11.848640

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '31054ccb0722'
down_revision: Union[str, None] = '3ef25a559d5b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'ga_generation_metrics',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('generation_number', sa.Integer(), nullable=False),
        sa.Column('best_fitness', sa.Float(), nullable=False),
        sa.Column('avg_fitness', sa.Float(), nullable=False),
        sa.Column('population_diversity', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['ga_experiment_runs.id']),
    )


def downgrade() -> None:
    op.drop_table('ga_generation_metrics')
