"""验证 ChnSentiCorp 数据集完整性。

检查三个 parquet 划分的行数是否符合预期(9600 / 1200 / 1200),
列名是否为 (text, label),并打印标签分布与少量样本。

在项目根目录执行:
    uv run python src/scripts/verify_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "ChnSentiCorp" / "parquet"

EXPECTED_ROWS = {"train": 9600, "validation": 1200, "test": 1200}
EXPECTED_COLS = {"text", "label"}


def check_split(split: str, expected_rows: int) -> bool:
    path = DATA_DIR / f"{split}.parquet"
    df = pd.read_parquet(path)
    ok_rows = len(df) == expected_rows
    ok_cols = set(df.columns) == EXPECTED_COLS

    print(f"\n=== {split} ===")
    print(f"  路径: {path.relative_to(PROJECT_ROOT)}")
    print(f"  行数: {len(df)} (预期 {expected_rows}) -- "
          f"{'OK' if ok_rows else '不匹配'}")
    print(f"  列名: {list(df.columns)} -- "
          f"{'OK' if ok_cols else '不匹配'}")
    print(f"  标签分布: {df['label'].value_counts().to_dict()}")
    print(f"  样本预览:")
    for _, row in df.head(2).iterrows():
        text_preview = str(row["text"])[:60].replace("\n", " ")
        print(f"    [label={row['label']}] {text_preview}...")

    return ok_rows and ok_cols


def main() -> None:
    all_ok = True
    for split, expected in EXPECTED_ROWS.items():
        all_ok &= check_split(split, expected)

    print("\n" + ("=" * 40))
    print("全部检查通过" if all_ok else "存在不匹配,需要重新下载")


if __name__ == "__main__":
    main()
