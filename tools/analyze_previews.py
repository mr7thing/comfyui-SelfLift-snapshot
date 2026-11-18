"""Is the fast preview a fair picture of the run?

Every draft run in the sweep leaves two videos behind: the node's own preview
(the small `taeh3` decoder, at the prefix's own resolution) and the very same
latent decoded by the real H3 video VAE through the ordinary decode chain. The
preview is only useful for judging a clip if the two agree.

The script also measures how sharp each of them is, because a preview that is
blurrier than the real decode hides detail rather than showing it.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/analyze_previews.py
"""
import json
import os
import sys

import numpy as np

WORK = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_test\output"


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


def sharpness(stack):
    """Mean gradient energy: how much detail the picture actually carries."""
    values = stack.astype(np.float32)
    dx = np.abs(np.diff(values, axis=2)).mean()
    dy = np.abs(np.diff(values, axis=1)).mean()
    return float((dx + dy) / 2)


def motion(stack):
    return float(np.abs(np.diff(stack.astype(np.float32), axis=0)).mean())


def compare(a, b):
    """Agreement between two clips of the same length."""
    count = min(len(a), len(b))
    a_f = a[:count].astype(np.float32)
    b_f = b[:count].astype(np.float32)
    mae = np.abs(a_f - b_f).mean(axis=(1, 2, 3))
    flat_a = a_f.reshape(count, -1)
    flat_b = b_f.reshape(count, -1)
    corr = np.array([float(np.corrcoef(flat_a[i], flat_b[i])[0, 1]) for i in range(count)])
    return {"mae": float(mae.mean()), "corr": float(corr.mean()), "worst": float(corr.min())}


def main():
    path = os.path.join(WORK, "preview_sweep.json")
    if not os.path.exists(path):
        print(f"  no sweep results in {path}")
        return 1
    with open(path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    drafts = [e for e in entries if e.get("preview") and e.get("clip") and e["spec"][0] == "draft"]
    fulls = [e for e in entries if e.get("clip") and e["spec"][0] == "full"]

    print("\n  the fast preview against the real decoder, same latent")
    print(f"  {'target / scale':<22}{'preview':<12}{'taeh3 sharp':>12}{'real sharp':>12}"
          f"{'agree':>9}{'corr':>8}{'s':>7}")
    for entry in drafts:
        preview = read(entry["preview"])
        real = read(entry["clip"])
        height, width = preview.shape[1:3]
        if real.shape[1:3] != (height, width):
            real = resize(real, width, height)
        stats = compare(preview, real)
        mode, megapixels, scale, steps = entry["spec"]
        print(f"  {str(megapixels) + 'MP / ' + str(scale):<22}"
              f"{str(width) + 'x' + str(height):<12}"
              f"{sharpness(preview):>12.2f}{sharpness(real):>12.2f}"
              f"{stats['mae']:>9.1f}{stats['corr']:>8.3f}{entry['seconds']:>7.0f}")

    print("\n  the finished clips")
    print(f"  {'target / scale':<22}{'clip':<12}{'sharp':>9}{'motion':>9}{'s':>7}")
    for entry in fulls:
        clip = read(entry["clip"])
        mode, megapixels, scale, steps = entry["spec"]
        print(f"  {str(megapixels) + 'MP / ' + str(scale):<22}"
              f"{str(clip.shape[2]) + 'x' + str(clip.shape[1]):<12}"
              f"{sharpness(clip):>9.2f}{motion(clip):>9.2f}{entry['seconds']:>7.0f}")

    print("\n  how far the preview drifts from the clip it predicts")
    print(f"  {'draft':<22}{'final':<22}{'pixel diff':>12}{'corr':>8}")
    for draft in drafts:
        for full in fulls:
            if (draft["spec"][1], draft["spec"][2]) != (full["spec"][1], full["spec"][2]):
                continue
            preview = read(draft["preview"])
            clip = read(full["clip"])
            height, width = preview.shape[1:3]
            small = resize(clip, width, height)
            stats = compare(preview, small)
            label = f"{draft['spec'][1]}MP / {draft['spec'][2]} ({draft['spec'][3]} steps)"
            final = f"{full['spec'][1]}MP / {full['spec'][2]} ({full['spec'][3]} steps)"
            print(f"  {label:<22}{final:<22}{stats['mae']:>12.1f}{stats['corr']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
