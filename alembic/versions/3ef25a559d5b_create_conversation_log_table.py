"""create_conversation_log_table

Revision ID: 3ef25a559d5b
Revises: cbe97010a189
Create Date: 2025-06-16 18:20:24.430147

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ef25a559d5b'
down_revision: Union[str, None] = 'cbe97010a189'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
