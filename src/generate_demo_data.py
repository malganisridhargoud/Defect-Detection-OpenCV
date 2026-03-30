"""
generate_demo_data.py
---------------------
Generates synthetic training images for demonstration when
MVTec dataset is not available. Creates realistic-looking
surface images with and without defects.
"""

import cv2
import numpy as np
import os
import random


def make_good_surface(size=224) -> np.ndarray:
    """Generate a clean, uniform surface texture."""
    # Base uniform gray surface
    img = np.ones((size, size), dtype=np.uint8) * random.randint(160, 200)

    # Add subtle grain texture
    noise = np.random.normal(0, 4, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add faint surface pattern (like brushed metal)
    for i in range(0, size, random.randint(8, 15)):
        cv2.line(img, (0, i), (size, i + random.randint(-2, 2)),
                 int(img[i, size//2]) - random.randint(2, 6), 1)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_defective_surface(size=224, defect_type=None) -> np.ndarray:
    """Generate surface with a synthetic defect."""
    img = make_good_surface(size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    defect_types = ["scratch", "crack", "hole", "stain", "dent"]
    if defect_type is None:
        defect_type = random.choice(defect_types)

    if defect_type == "scratch":
        # Thin elongated line
        x1, y1 = random.randint(20, 80), random.randint(20, 80)
        x2, y2 = x1 + random.randint(60, 140), y1 + random.randint(-20, 20)
        cv2.line(img, (x1, y1), (x2, y2), (30, 30, 30), random.randint(1, 3))

    elif defect_type == "crack":
        # Jagged multi-segment crack
        pts = [(random.randint(30, 100), random.randint(80, 140))]
        for _ in range(random.randint(4, 8)):
            last = pts[-1]
            pts.append((last[0] + random.randint(10, 25),
                        last[1] + random.randint(-15, 15)))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], (20, 20, 20), random.randint(1, 2))

    elif defect_type == "hole":
        # Dark circular pit
        cx, cy = random.randint(60, 164), random.randint(60, 164)
        r = random.randint(6, 18)
        cv2.circle(img, (cx, cy), r, (15, 15, 15), -1)
        cv2.circle(img, (cx, cy), r + 1, (80, 80, 80), 1)

    elif defect_type == "stain":
        # Irregular dark blob
        pts = np.array([
            [random.randint(50, 174), random.randint(50, 174)]
            for _ in range(random.randint(5, 10))
        ], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(img, [hull], (random.randint(40, 90),) * 3)

    elif defect_type == "dent":
        # Elliptical depression with shadow effect
        cx, cy = random.randint(60, 164), random.randint(60, 164)
        a, b = random.randint(15, 35), random.randint(10, 25)
        cv2.ellipse(img, (cx, cy), (a, b), random.randint(0, 180),
                    0, 360, (50, 50, 60), -1)
        cv2.ellipse(img, (cx - 3, cy - 3), (a, b), random.randint(0, 180),
                    0, 180, (200, 200, 210), 1)

    return img


def generate_dataset(base_dir: str, n_good: int = 150, n_defective: int = 150):
    """Generate full synthetic dataset."""
    splits = {
        "train/good": (int(n_good * 0.8), make_good_surface),
        "train/defective": (int(n_defective * 0.8), make_defective_surface),
        "test/good": (int(n_good * 0.2), make_good_surface),
        "test/defective": (int(n_defective * 0.2), make_defective_surface),
    }

    total = 0
    for split, (count, fn) in splits.items():
        folder = os.path.join(base_dir, split)
        os.makedirs(folder, exist_ok=True)
        for i in range(count):
            img = fn()
            cv2.imwrite(os.path.join(folder, f"img_{i:04d}.png"), img)
            total += 1

    print(f"Generated {total} synthetic images in {base_dir}")
    return total


if __name__ == "__main__":
    generate_dataset("data")
