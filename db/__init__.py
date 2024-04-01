from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from settings import settings

Base = declarative_base()
engine = create_engine(settings.sql_url)

Session = sessionmaker(bind=engine)
session = Session()
