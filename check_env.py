import os, sys
from pathlib import Path

print("Python 版本  :", sys.version.split()[0])
print("工作目录     :", Path.cwd())

try:
    import tushare, pyarrow
    from dotenv import load_dotenv
    print("依赖         : OK")
except Exception as e:
    print("依赖缺失     :", e)
    sys.exit(1)

load_dotenv()

print(".env 存在    :", Path(".env").exists())
print("config.py 存在:", Path("config.py").exists())
print("token 长度   :", len(os.getenv("TUSHARE_TOKEN", "")))

import config
print("config 路径  :", config.__file__)
print("config 变量  :", [x for x in dir(config) if not x.startswith("_")])
