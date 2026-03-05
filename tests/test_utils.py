import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import db_connect, get_database_url


class UtilsTests(unittest.TestCase):
    def test_get_database_url_raises_when_missing(self):
        previous = os.environ.pop("DATABASE_URL", None)
        try:
            with self.assertRaises(ValueError):
                get_database_url()
        finally:
            if previous is not None:
                os.environ["DATABASE_URL"] = previous

    def test_db_connect_returns_engine(self):
        previous = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
        try:
            engine = db_connect()
            self.assertEqual(engine.dialect.name, "sqlite")
        finally:
            if previous is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous


if __name__ == "__main__":
    unittest.main()
