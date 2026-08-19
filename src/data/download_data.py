"""
src/data/download_data.py

Dataset download script for the IEEE-CIS Fraud Detection competition.

Steps:
  1. Check ~/.kaggle/kaggle.json credentials
  2. Download IEEE-CIS dataset via Kaggle API
  3. Extract CSV files to data/raw/
  4. Print SHA256 checksums for reproducibility

Usage:
    python src/data/download_data.py
    python src/data/download_data.py --skip-if-exists  # no-op if CSVs present
"""

import argparse
import hashlib
import json
import logging
import os
import pathlib
import shutil
import sys
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
KAGGLE_CONFIG_PATH = pathlib.Path.home() / ".kaggle" / "kaggle.json"
COMPETITION = "ieee-fraud-detection"
EXPECTED_FILES = ["train_transaction.csv", "train_identity.csv"]


def check_kaggle_credentials() -> None:
    """Assert that ~/.kaggle/kaggle.json exists and contains required keys."""
    if not KAGGLE_CONFIG_PATH.exists():
        logger.error(
            f"Kaggle credentials not found at {KAGGLE_CONFIG_PATH}. "
            "Please create it with your username and API key. "
            "Download from: https://www.kaggle.com/settings/account → 'Create New API Token'."
        )
        sys.exit(1)

    with open(KAGGLE_CONFIG_PATH) as f:
        creds = json.load(f)

    missing = [k for k in ("username", "key") if k not in creds]
    if missing:
        logger.error(f"kaggle.json is missing keys: {missing}")
        sys.exit(1)

    logger.info(f"Kaggle credentials found for user: {creds['username']}")


def check_existing_files() -> bool:
    """Return True if all expected CSV files are already present in data/raw/."""
    present = [f for f in EXPECTED_FILES if (DATA_RAW_DIR / f).exists()]
    if len(present) == len(EXPECTED_FILES):
        logger.info("All dataset files already present in data/raw/ — skipping download.")
        return True
    return False


def download_dataset() -> None:
    """Download the IEEE-CIS competition data using the Kaggle API."""
    import kaggle  # noqa: F401 — validates kaggle is installed

    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading competition data: {COMPETITION}")
    api.competition_download_files(
        competition=COMPETITION,
        path=str(DATA_RAW_DIR),
        quiet=False,
    )
    logger.info("Download complete.")



def extract_zip() -> None:
    """Extract downloaded zip archive to data/raw/ and remove the zip."""
    zip_path = DATA_RAW_DIR / f"{COMPETITION}.zip"

    if not zip_path.exists():
        # Some versions download individual file zips
        zip_files = list(DATA_RAW_DIR.glob("*.zip"))
        if not zip_files:
            logger.warning("No zip files found in data/raw/ — may already be extracted.")
            return
        zip_path = zip_files[0]

    logger.info(f"Extracting {zip_path.name} → {DATA_RAW_DIR}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_RAW_DIR)

    # Some Kaggle downloads contain nested zips for individual files
    for nested_zip in DATA_RAW_DIR.glob("*.zip"):
        logger.info(f"Extracting nested zip: {nested_zip.name}")
        with zipfile.ZipFile(nested_zip, "r") as zf:
            zf.extractall(DATA_RAW_DIR)
        nested_zip.unlink()
        logger.info(f"Removed: {nested_zip.name}")

    logger.info("Extraction complete.")


def compute_checksums() -> None:
    """Print SHA256 checksums of all CSV files in data/raw/ for reproducibility."""
    logger.info("SHA256 checksums for reproducibility verification:")
    for filename in EXPECTED_FILES:
        filepath = DATA_RAW_DIR / filename
        if filepath.exists():
            sha256 = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  {filename}: SHA256={sha256.hexdigest()}, Size={size_mb:.1f}MB")
        else:
            logger.warning(f"  {filename}: NOT FOUND")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download IEEE-CIS Fraud Detection dataset from Kaggle."
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip download if CSV files are already present in data/raw/.",
    )
    args = parser.parse_args()

    check_kaggle_credentials()

    if args.skip_if_exists and check_existing_files():
        compute_checksums()
        return

    download_dataset()
    extract_zip()
    compute_checksums()

    # Verify expected files exist after download
    missing = [f for f in EXPECTED_FILES if not (DATA_RAW_DIR / f).exists()]
    if missing:
        logger.error(f"Download completed but files are missing: {missing}")
        logger.error("Check Kaggle competition acceptance at: https://www.kaggle.com/competitions/ieee-fraud-detection")
        sys.exit(1)

    logger.info("✓ All dataset files verified in data/raw/")


if __name__ == "__main__":
    main()
