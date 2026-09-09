#!/usr/bin/env python3
"""
FinOps-Security-Agent — NeurIPS 2022 Dataset Downloader Utility
Downloads and prepares the 1,000,000-row Kaggle NeurIPS 2022 Bank Account Fraud dataset (Base.csv).
"""

import os
import urllib.request
from src.logger import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "BAF_NeurIPS_2022_dataset")
TARGET_FILE = os.path.join(DATASET_DIR, "Base.csv")

# Public reference mirror URL for Kaggle NeurIPS 2022 BAF dataset
PUBLIC_DATASET_URL = "https://raw.githubusercontent.com/Feedzai/bank-account-fraud/main/data/Base.csv"

def download_baf_dataset():
    """Downloads Base.csv if not present locally."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(TARGET_FILE):
        size_mb = os.path.getsize(TARGET_FILE) / (1024 * 1024)
        logger.info(f"Dataset already present at {TARGET_FILE} ({size_mb:.2f} MB).")
        return TARGET_FILE

    logger.info(f"Downloading Kaggle NeurIPS 2022 Dataset from mirror...")
    try:
        urllib.request.urlretrieve(PUBLIC_DATASET_URL, TARGET_FILE)
        size_mb = os.path.getsize(TARGET_FILE) / (1024 * 1024)
        logger.info(f"Dataset successfully downloaded to {TARGET_FILE} ({size_mb:.2f} MB).")
        return TARGET_FILE
    except Exception as e:
        logger.warning(f"Could not download dataset from public mirror: {e}")
        logger.info("Using local stratified dataset fallback.")
        return None

if __name__ == "__main__":
    download_baf_dataset()
