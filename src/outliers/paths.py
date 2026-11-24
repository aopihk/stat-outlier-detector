from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.toml"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "nba.db"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
