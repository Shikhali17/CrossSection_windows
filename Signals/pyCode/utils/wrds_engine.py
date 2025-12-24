import os
import urllib.parse
from sqlalchemy import create_engine
from dotenv import load_dotenv

def get_wrds_engine():
    # Load .env reliably regardless of working directory
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

    user = os.getenv("WRDS_USERNAME")
    pw = os.getenv("WRDS_PASSWORD")
    if not user or not pw:
        raise ValueError("Missing WRDS_USERNAME or WRDS_PASSWORD in environment variables")

    pw_enc = urllib.parse.quote_plus(pw)

    host = "wrds-pgdata.wharton.upenn.edu"
    port = 9737
    db = "wrds"

    # Force TCP + SSL (important on Windows)
    url = f"postgresql+psycopg2://{user}:{pw_enc}@{host}:{port}/{db}?sslmode=require"
    return create_engine(url)