import os

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

DATABASE_URL = os.environ["DATABASE_URL"]

# Railway/Supabase entregan "postgresql://..." pero SQLAlchemy async
# necesita el driver asyncpg: "postgresql+asyncpg://..."
if DATABASE_URL.startswith(("postgresql://", "postgres://")):
    DATABASE_URL = DATABASE_URL.replace("://", "+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={"statement_cache_size": 0},  # obligatorio con el pooler de Supabase
)

async_session = async_sessionmaker(engine, expire_on_commit=False)
