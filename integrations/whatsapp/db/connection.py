import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

DATABASE_URL = os.environ["DATABASE_URL"]

# Railway/Supabase entregan "postgresql://..." o "postgres://...",
# pero SQLAlchemy async necesita el driver asyncpg:
if DATABASE_URL.startswith(("postgresql://", "postgres://")):
    DATABASE_URL = DATABASE_URL.replace("://", "+asyncpg://", 1)


def _translate_libpq_params(url: str) -> str:
    """
    Los query params tipo libpq (sslmode, channel_binding, sslcert...)
    rompen asyncpg porque los recibe como kwargs de connect().
    Traducimos sslmode=require → ssl=require (asyncpg sí lo entiende)
    y eliminamos el resto.
    """
    LIBPQ_ONLY = {"sslcert", "sslkey", "sslrootcert", "channel_binding"}

    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)

    fixed = []
    for key, value in query:
        if key == "sslmode":          # traducir al parámetro que asyncpg entiende
            fixed.append(("ssl", value))
        elif key in LIBPQ_ONLY:       # descartar (no aplican a asyncpg)
            continue
        else:
            fixed.append((key, value))

    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(fixed), parts.fragment))


DATABASE_URL = _translate_libpq_params(DATABASE_URL)

engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={"statement_cache_size": 0},  # obligatorio con el pooler de Supabase
)

async_session = async_sessionmaker(engine, expire_on_commit=False)
