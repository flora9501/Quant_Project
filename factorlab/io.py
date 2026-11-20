# factorlab/io.py
"""原子写。全项目唯一实现。

parquet / json / text 三种，语义一致：
  同目录临时文件 → 写入 → fsync → os.replace

为什么必须同目录：os.replace 只在同一文件系统内是原子的，
跨文件系统的 rename 会退化成 copy + delete，失去原子性。
"""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def _commit(tmp: Path, path: Path) -> None:
    """fsync 数据 → chmod → 原子替换 → 尽力 fsync 目录项。"""
    with open(tmp, "rb+") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(tmp, 0o644)          # mkstemp 默认 0600
    os.replace(tmp, path)
    try:                          # 目录项也落盘；macOS 部分文件系统不支持，忽略
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def _mkstemp(path: Path):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    return Path(tmp)


def atomic_write_parquet(df: pd.DataFrame, path, **kw) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _mkstemp(path)
    try:
        df.to_parquet(tmp, **kw)
        _commit(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(obj, path, **kw) -> None:
    kw.setdefault("ensure_ascii", False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _mkstemp(path)
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, **kw)
        _commit(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(text: str, path, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _mkstemp(path)
    try:
        with open(tmp, "w", encoding=encoding) as fh:
            fh.write(text)
        _commit(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
