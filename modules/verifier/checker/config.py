import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / "fist.env"
load_dotenv(ENV_PATH, override=True)

CHECKER_MODEL = os.getenv("CHECKER_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
print("API key loaded:", bool(OPENAI_API_KEY))
print("Model:", CHECKER_MODEL)