#!/usr/bin/env python
# coding: utf-8

"""
src/preprocess.py
BUSI dataset preprocessing and curation script using official mapping.
Reference: Aumente-Maestro et al. (2025).
"""

import os
import re
from pathlib import Path
from typing import Dict, Set
import pandas as pd

DATA_DIR = Path("data")
MAPPING_CSV = DATA_DIR / "mapping_curated_BUSI.csv"
CLASSES = ["benign", "malignant", "normal"]


def extract_id_from_filename(filename: str) -> int:
    match = re.search(r"\((\d+)\)", filename)
    return int(match.group(1)) if match else -1


def clean_by_curated_mapping(data_dir: Path, mapping_csv: Path) -> None:
    print("=" * 70)
    print("[INFO] Starting Dataset Curation (Aumente-Maestro et al., 2025)")
    print(f"[INFO] Data directory : {data_dir.resolve()}")
    print(f"[INFO] Mapping file   : {mapping_csv.resolve()}")
    print("=" * 70)

    df_mapping = pd.read_csv(mapping_csv, sep=";")
    curated_ids: Dict[str, Set[int]] = {
        cls: set(df_mapping[df_mapping["class"] == cls]["id"].astype(int))
        for cls in CLASSES
    }

    total_scanned = 0
    total_images_deleted = 0
    total_masks_deleted = 0
    total_kept = 0

    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"[WARNING] Class directory not found: {cls_dir}")
            continue

        all_pngs = sorted(cls_dir.glob("*.png"))
        image_files = [f for f in all_pngs if "_mask" not in f.name.lower()]

        cls_scanned = len(image_files)
        cls_deleted = 0
        cls_masks_del = 0
        cls_kept = 0
        total_scanned += cls_scanned

        print(f"\n[INFO] Processing class: [{cls.upper()}] (Found {cls_scanned} images)")

        for img_path in image_files:
            img_id = extract_id_from_filename(img_path.name)

            if img_id not in curated_ids[cls]:
                cls_deleted += 1
                total_images_deleted += 1
                img_path.unlink()

                # Delete associated masks
                for mask_file in cls_dir.glob(f"{img_path.stem}_mask*.png"):
                    mask_file.unlink()
                    cls_masks_del += 1
                    total_masks_deleted += 1
            else:
                cls_kept += 1
                total_kept += 1

        print(f"  [RESULT] Class [{cls}]: Kept {cls_kept} | Removed {cls_deleted} images | Removed {cls_masks_del} masks")

    print("\n" + "=" * 70)
    print("[SUMMARY] Preprocessing & Curation Summary:")
    print(f" - Initial images scanned  : {total_scanned}")
    print(f" - Removed duplicate/noise : {total_images_deleted}")
    print(f" - Removed masks           : {total_masks_deleted}")
    print(f" - Curated images kept     : {total_kept}")
    print("=" * 70)
    print("[INFO] Dataset preprocessing completed successfully.\n")


def main():
    if not MAPPING_CSV.exists():
        print(f"[ERROR] Mapping file '{MAPPING_CSV}' not found!")
        return
    clean_by_curated_mapping(DATA_DIR, MAPPING_CSV)


if __name__ == "__main__":
    main()