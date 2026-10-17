from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from settings import settings

Base = declarative_base()
engine = create_engine(settings.sql_url)
Session = sessionmaker(bind=engine)
session = Session()


def init_db():
    from db import models  # noqa: F401 -- register tables before create_all

    if engine.url.get_backend_name() == "sqlite" and engine.url.database not in (
        None,
        "",
        ":memory:",
    ):
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
