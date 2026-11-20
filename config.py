import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()

CHECK_DATE = "20240102"     # 验收日期
INDEX_CODE = "000852.SH"    # 中证 1000

DATA_DIR = Path("data")
CHECK_DIR = DATA_DIR / "_check"
CHECK_DIR.mkdir(parents=True, exist_ok=True)
