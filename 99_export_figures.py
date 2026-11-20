# 99_export_figures.py
from pathlib import Path
import shutil

MAP = {
    "nav.png":                "nav.png",
    "excess_nav_compare.png": "excess_nav_compare.png",
    "layers.png":             "layers.png",
    "ic_decay.png":           "ic_decay.png",
}

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
found = {p.name: p for p in Path("data").rglob("*.png")}

for src_name, dst_name in MAP.items():
    if src_name in found:
        shutil.copy2(found[src_name], OUT / dst_name)
        print(f"✅ {found[src_name]} -> figures/{dst_name}")
    else:
        print(f"❌ 没找到 {src_name}")
