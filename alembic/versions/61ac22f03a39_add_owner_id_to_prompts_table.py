"""add_owner_id_to_prompts_table

Revision ID: 61ac22f03a39
Revises: f0d711efe1a3
Create Date: 2025-06-16 14:29:11.557011

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '61ac22f03a39'
down_revision: Union[str, None] = 'f0d711efe1a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('prompts', schema=None) as batch_op:
        batch_op.add_column(sa.Column('owner_id', sa.Integer(), nullable=False))
        batch_op.create_foreign_key(
            'fk_prompts_owner_id_users',
            'users',
            ['owner_id'],
            ['id']
        )


def downgrade() -> None:
    with op.batch_alter_table('prompts', schema=None) as batch_op:
        batch_op.drop_constraint('fk_prompts_owner_id_users', type_='foreignkey')
        batch_op.drop_column('owner_id')
