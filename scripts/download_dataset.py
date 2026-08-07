"""Download CORD-v2 dataset from Hugging Face and save to data/raw/."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO_ID = "naver-clova-ix/cord-v2"
SPLITS = {
    "train": [
        "data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
        "data/train-00001-of-00004-7dbbe248962764c5.parquet",
        "data/train-00002-of-00004-688fe1305a55e5cc.parquet",
        "data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
    ],
    "validation": [
        "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    ],
    "test": [
        "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
    ],
}


def main():
    import pandas as pd
    from PIL import Image
    import io

    output_dir = Path("data/raw")

    for split_name, parquet_files in SPLITS.items():
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        annotations_dir = split_dir / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading {split_name}...")

        idx = 0
        for pf in parquet_files:
            local_path = hf_hub_download(
                repo_id=REPO_ID, filename=pf, repo_type="dataset"
            )
            df = pd.read_parquet(local_path)

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}"):
                # Extract image
                img_bytes = row["image"]["bytes"]
                image = Image.open(io.BytesIO(img_bytes))
                image.save(images_dir / f"{idx:04d}.png")

                # Extract annotation
                ground_truth = json.loads(row["ground_truth"])
                ann_path = annotations_dir / f"{idx:04d}.json"
                ann_path.write_text(
                    json.dumps(ground_truth, indent=2, ensure_ascii=False)
                )
                idx += 1

        print(f"  Saved {idx} samples to {split_dir}")

    print(f"\nDone! Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
