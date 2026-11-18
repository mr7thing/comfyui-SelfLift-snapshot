"""The step split under test: how many steps stay low, how many run high.

Every clip is measured against the same reference -- the two/two split of the
four-step schedule -- at the reference's own resolution, so the numbers are
about content and motion rather than about the resolution gap.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/compare_splits.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_draft_final import motion, read, resize  # noqa: E402

OUT = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_refs\splits"
REFS = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_refs"
NOW = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\video"

REFERENCE = ("2 low + 2 high  (4 steps)", os.path.join(REFS, "fast_0p4mp_4nfe.mp4"))
CLIPS = [
    ("3 low + 1 high  (4 steps)", os.path.join(NOW, "split_3_1_00001_.mp4")),
    ("4 low + 2 high  (6 steps)", os.path.join(NOW, "split_4_2_00001_.mp4")),
    ("5 low + 1 high  (6 steps)", os.path.join(NOW, "split_5_1_00001_.mp4")),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    from PIL import Image

    label_ref, path_ref = REFERENCE
    if not os.path.exists(path_ref):
        print("  the reference clip is missing")
        return 1
    ref = read(path_ref)
    height, width = ref.shape[1:3]
    ref_small = resize(ref, width, height).astype(np.float32)
    ref_motion = motion(ref)

    print(f"  reference: {label_ref}  {ref.shape[2]}x{ref.shape[1]}  motion {ref_motion:.3f}")
    print(f"  {'split':<26} {'pixels vs ref':>15} {'corr':>7} {'worst':>7} {'motion':>8} {'drift tail':>11}")
    for label, path in CLIPS:
        if not os.path.exists(path):
            print(f"  {label:<26} missing")
            continue
        clip = read(path)
        count = min(len(clip), len(ref_small))
        small = resize(clip[:count], width, height).astype(np.float32)
        other = ref_small[:count]
        mae = np.abs(small - other).mean(axis=(1, 2, 3))
        flat_a = small.reshape(count, -1)
        flat_b = other.reshape(count, -1)
        corr = np.array([float(np.corrcoef(flat_a[i], flat_b[i])[0, 1]) for i in range(count)])
        clip_motion = motion(clip[:count])
        third = count // 3
        print(f"  {label:<26} {mae.mean():10.2f}/255 {corr.mean():7.3f} {corr.min():7.3f} "
              f"{clip_motion:8.3f} {mae[-third:].mean():11.2f}")

        for index in (0, count // 2, count - 1):
            difference = np.abs(small[index] - other[index]).mean(axis=2)
            sheet = np.concatenate([
                other[index].astype(np.uint8),
                small[index].astype(np.uint8),
                np.stack([np.clip(difference * 8, 0, 255).astype(np.uint8)] * 3, axis=2),
            ], axis=1).astype(np.uint8)
            name = f"{os.path.basename(path).split('_')[1]}_{index:03d}.png"
            Image.fromarray(sheet).save(os.path.join(OUT, name))
    print(f"  sheets in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
