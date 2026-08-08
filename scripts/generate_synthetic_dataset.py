"""Generate synthetic degraded images for CNN quality assessment training."""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with given sigma."""
    ksize = int(6 * sigma + 1) | 1  # ensure odd
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_motion_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply horizontal motion blur."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle (degrees)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))


def apply_gaussian_noise(image: np.ndarray, std: float = 25.0) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_salt_pepper_noise(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """Add salt and pepper noise."""
    result = image.copy()
    # Salt
    n_salt = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, n_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:
        result[coords[0], coords[1], :] = 255
    else:
        result[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i, n_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:
        result[coords[0], coords[1], :] = 0
    else:
        result[coords[0], coords[1]] = 0
    return result


def apply_downscale(image: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """Downscale then upscale to simulate low resolution."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_jpeg_compression(image: np.ndarray, quality: int = 10) -> np.ndarray:
    """Simulate heavy JPEG compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR if len(image.shape) == 3 else cv2.IMREAD_GRAYSCALE)


def apply_irregular_lighting(image: np.ndarray) -> np.ndarray:
    """Apply irregular lighting gradient."""
    h, w = image.shape[:2]
    # Create a random gradient
    x = np.random.randint(0, w)
    y = np.random.randint(0, h)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2).astype(np.float32)
    dist = dist / dist.max()
    # Darken based on distance
    factor = 0.4 + 0.6 * (1.0 - dist)

    if len(image.shape) == 3:
        factor = factor[:, :, np.newaxis]

    result = (image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    return result


# All degradation functions with parameters
DEGRADATIONS = {
    "blur_light": lambda img: apply_gaussian_blur(img, sigma=1.0),
    "blur_medium": lambda img: apply_gaussian_blur(img, sigma=3.0),
    "blur_heavy": lambda img: apply_gaussian_blur(img, sigma=5.0),
    "motion_blur": lambda img: apply_motion_blur(img, kernel_size=15),
    "rotation_5": lambda img: apply_rotation(img, angle=random.choice([-5, 5])),
    "rotation_10": lambda img: apply_rotation(img, angle=random.choice([-10, 10])),
    "rotation_15": lambda img: apply_rotation(img, angle=random.choice([-15, 15])),
    "noise_light": lambda img: apply_gaussian_noise(img, std=15),
    "noise_heavy": lambda img: apply_gaussian_noise(img, std=40),
    "salt_pepper": lambda img: apply_salt_pepper_noise(img, amount=0.03),
    "low_res_25": lambda img: apply_downscale(img, factor=0.25),
    "low_res_50": lambda img: apply_downscale(img, factor=0.50),
    "jpeg_q10": lambda img: apply_jpeg_compression(img, quality=10),
    "jpeg_q20": lambda img: apply_jpeg_compression(img, quality=20),
    "jpeg_q30": lambda img: apply_jpeg_compression(img, quality=30),
    "lighting": lambda img: apply_irregular_lighting(img),
}


def generate_degradations(
    input_dir: Path,
    output_dir: Path,
    n_degradations_per_image: int = 5,
    seed: int = 42,
):
    """Generate degraded images from original dataset.

    Args:
        input_dir: Directory with original images.
        output_dir: Directory to save degraded images.
        n_degradations_per_image: Number of degradations to apply per image.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "original").mkdir(exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))
    degradation_names = list(DEGRADATIONS.keys())

    print(f"Found {len(image_paths)} images")
    print(f"Generating {n_degradations_per_image} degradations per image")
    print(f"Available degradations: {len(degradation_names)}")
    print(f"Total output: {len(image_paths)} originals + {len(image_paths) * n_degradations_per_image} degraded")

    metadata = []

    for img_path in tqdm(image_paths, desc="Generating"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        stem = img_path.stem

        # Save original
        cv2.imwrite(str(output_dir / "original" / f"{stem}.png"), img)

        # Apply random degradations
        selected = random.sample(degradation_names, min(n_degradations_per_image, len(degradation_names)))

        for deg_name in selected:
            deg_dir = output_dir / deg_name
            deg_dir.mkdir(exist_ok=True)

            degraded = DEGRADATIONS[deg_name](img)
            out_path = deg_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), degraded)

            metadata.append({
                "original": str(img_path.name),
                "degradation": deg_name,
                "output": str(out_path.relative_to(output_dir)),
            })

    # Save metadata
    import json
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nDone! Metadata saved to {meta_path}")
    print(f"Total degraded images: {len(metadata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic degraded images")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/train/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-degradations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_degradations(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_degradations_per_image=args.n_degradations,
        seed=args.seed,
    )
