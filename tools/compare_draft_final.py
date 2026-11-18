"""How far the preview drifts from the clip it predicts.

The draft samples the low-resolution prefix of a run; the full run lifts that
prefix and finishes the schedule at the target resolution. Both start from the
same noise, so the question is how much the high-resolution continuation moves
the picture away from the preview.

Both clips are compared at the preview's own resolution: the full run is
downscaled to it first, so the numbers are about content and motion, not about
the resolution gap.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/compare_draft_final.py
"""
import os
import sys

import numpy as np

OUT = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_refs\compare"
REFS = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_refs"
NOW = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\video"

PAIRS = [
    ("draft (3/4 steps low-res)", os.path.join(NOW, "draft_00001_.mp4"),
     "final 4 NFE (lift + 2 high-res steps)", os.path.join(REFS, "fast_0p4mp_4nfe.mp4")),
    ("draftfull (4/4 steps low-res)", os.path.join(NOW, "draftfull_00001_.mp4"),
     "final 4 NFE (lift + 2 high-res steps)", os.path.join(REFS, "fast_0p4mp_4nfe.mp4")),
]


def read(path):
    import av

    with av.open(path) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    return np.stack(frames)


def resize(stack, width, height):
    from PIL import Image

    out = np.empty((len(stack), height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(stack):
        out[index] = np.asarray(Image.fromarray(frame).resize((width, height), Image.BICUBIC))
    return out


def motion(stack):
    """Mean frame-to-frame change: how much the picture moves over time."""
    return float(np.abs(np.diff(stack.astype(np.float32), axis=0)).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    from PIL import Image

    for label_a, path_a, label_b, path_b in PAIRS:
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            print(f"  missing a clip for {label_a}")
            continue
        a = read(path_a)
        b = read(path_b)
        height, width = a.shape[1:3]
        b_small = resize(b, width, height)
        count = min(len(a), len(b_small))

        a_f = a[:count].astype(np.float32)
        b_f = b_small[:count].astype(np.float32)
        mae = np.abs(a_f - b_f).mean(axis=(1, 2, 3))
        flat_a = a_f.reshape(count, -1)
        flat_b = b_f.reshape(count, -1)
        corr = np.array([
            float(np.corrcoef(flat_a[i], flat_b[i])[0, 1]) for i in range(count)
        ])

        print(f"\n  {label_a}  vs  {label_b}")
        print(f"    sizes            {width}x{height}  vs  {b.shape[2]}x{b.shape[1]}")
        print(f"    pixel difference {mae.mean():6.2f} / 255   (per frame {mae.min():.1f}..{mae.max():.1f})")
        print(f"    frame correlation {corr.mean():.4f}  (worst frame {corr.min():.4f})")
        print(f"    motion           {motion(a[:count]):6.3f}  vs  {motion(b_small[:count]):6.3f}")
        third = count // 3
        print(f"    drift over time  first third {mae[:third].mean():5.2f}   "
              f"last third {mae[-third:].mean():5.2f}   "
              f"(correlation {corr[:third].mean():.3f} -> {corr[-third:].mean():.3f})")

        for index in (0, count // 2, count - 1):
            difference = np.abs(a_f[index] - b_f[index]).mean(axis=2)
            gain = min(255.0, difference.mean() * 8)
            diff_image = np.clip(difference * 8, 0, 255).astype(np.uint8)
            sheet = np.concatenate([a_f[index], b_f[index],
                                    np.stack([diff_image] * 3, axis=2)], axis=1).astype(np.uint8)
            name = f"{label_a.split()[0]}_{index:03d}.png"
            Image.fromarray(sheet).save(os.path.join(OUT, name))
        print(f"    sheets in {OUT}")


if __name__ == "__main__":
    main()
