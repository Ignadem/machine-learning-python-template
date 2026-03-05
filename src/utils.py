from dotenv import load_dotenv
from sqlalchemy import create_engine
import os

# load the .env file variables
load_dotenv()


def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set. Add it to your .env file.")
    return database_url


def db_connect():
    return create_engine(get_database_url(), pool_pre_ping=True)
