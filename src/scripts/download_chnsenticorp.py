"""Download the ChnSentiCorp dataset into this project.

Usage:
    python NLP/download_chnsenticorp.py

The script uses only the Python standard library and downloads the official
parquet files from Hugging Face:
    NLP/datasets/ChnSentiCorp/parquet/
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path


DATASET_NAME = "lansinuote/ChnSentiCorp"
SOURCE_URL = "https://huggingface.co/datasets/lansinuote/ChnSentiCorp"
BASE_URL = f"{SOURCE_URL}/resolve/main"

FILES = {
    "train": "data/train-00000-of-00001-02f200ca5f2a7868.parquet",
    "validation": "data/validation-00000-of-00001-405befbaa3bcf1a2.parquet",
    "test": "data/test-00000-of-00001-5372924f059fe767.parquet",
}

SPLIT_ROWS = {
    "train": 9600,
    "validation": 1200,
    "test": 1200,
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved {destination} ({destination.stat().st_size} bytes)")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "NLP" / "datasets" / "ChnSentiCorp"
    parquet_dir = output_dir / "parquet"

    for split, remote_path in FILES.items():
        destination = parquet_dir / f"{split}.parquet"
        download_file(f"{BASE_URL}/{remote_path}", destination)

    info = {
        "dataset_name": DATASET_NAME,
        "source": SOURCE_URL,
        "task": "Chinese sentiment classification",
        "format": "parquet",
        "columns": ["text", "label"],
        "label_mapping": {"0": "negative", "1": "positive"},
        "splits": SPLIT_ROWS,
        "files": {
            split: str((parquet_dir / f"{split}.parquet").relative_to(project_root))
            for split in FILES
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)

    print(f"Dataset info saved to {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
