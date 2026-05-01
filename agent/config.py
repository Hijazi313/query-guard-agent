import json
from pathlib import Path
import structlog
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

_ = load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

# Load cities dataset
_CITIES_PATH = Path(__file__).parent.parent / "cities.json"
with open(_CITIES_PATH) as f:
    _raw_cities = json.load(f)

CITY_DB = {k.lower(): v.lower() for k, v in _raw_cities.items()}
CITY_KEYS = list(CITY_DB.keys())

# Shared LLM Instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MAX_RETRY = 2
