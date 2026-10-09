"""Upload the competition dataset to the private Hugging Face repo.

Split out from the code/weights push because the 600 page-scale PNGs are 2.5 GB and take
several hours on a home uplink, while the code and checkpoints take under an hour. Run this
whenever the machine can stay awake and online — overnight is the intended use.

    .venv/bin/python scripts/push_dataset_to_hf.py

Resumable: `upload_large_folder` records per-file state in .cache/huggingface inside the folder,
so a killed run picks up where it stopped instead of re-uploading. Safe to run repeatedly.
"""
from huggingface_hub import HfApi

REPO = "siddhant20/task1"
ROOT = "/Users/siddhantparashar/projects/TASK1_CYBER"

if __name__ == "__main__":
    HfApi().upload_large_folder(
        repo_id=REPO,
        repo_type="model",
        folder_path=ROOT,
        # Everything except the dataset is already pushed; the zip is a duplicate of the
        # extracted folder and would double the transfer for nothing.
        allow_patterns=["Task1/**"],
        ignore_patterns=["*.zip", "**/.DS_Store"],
        num_workers=8,
        print_report_every=60,
    )
    print("dataset upload complete")
