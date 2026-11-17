"""Three clips, one table: what a cheap run buys and what it costs.

Compares the 0.16 MP full run against the 0.4 MP full run and the 24-step
baseline, and drops three frames of each next to each other.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/compare_three.py
"""
import os

import numpy as np

REFS = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_refs"
NOW = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_test\output\video"
FRAMES = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_refs\frames"

CLIPS = [
    ("0.16MP  4 NFE", os.path.join(NOW, "preview_00001_.mp4")),
    ("0.4MP   4 NFE", os.path.join(REFS, "fast_0p4mp_4nfe.mp4")),
    ("0.4MP  24 NFE", os.path.join(REFS, "base_0p4mp_24nfe.mp4")),
]


def read(path):
    import av

    with av.open(path) as container:
        kinds = {stream.type for stream in container.streams}
    with av.open(path) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    audio = []
    if "audio" in kinds:
        with av.open(path) as container:
            audio = [frame.to_ndarray() for frame in container.decode(audio=0)]
    pixels = np.stack([frame.astype(np.float32) for frame in frames])
    # Mean vertical gradient inside a frame: a rough stand-in for sharpness.
    detail = float(np.abs(np.diff(pixels, axis=1)).mean())
    samples = np.concatenate([chunk.reshape(-1) for chunk in audio]) if audio else np.zeros(1)
    return frames, float(np.sqrt((samples ** 2).mean())), detail


def main():
    os.makedirs(FRAMES, exist_ok=True)
    from PIL import Image

    print(f"  {'clip':<14} {'frames':>7} {'size':>12} {'detail':>8} {'audio rms':>10}")
    for label, path in CLIPS:
        if not os.path.exists(path):
            print(f"  {label:<14} missing")
            continue
        frames, rms, detail = read(path)
        height, width = frames[0].shape[:2]
        print(f"  {label:<14} {len(frames):>7} {f'{width}x{height}':>12} {detail:>8.2f} {rms:>10.4f}")
        middle = len(frames) // 2
        for index in (0, middle, len(frames) - 1):
            name = f"{label.split()[0]}_{index:03d}.png"
            Image.fromarray(frames[index]).save(os.path.join(FRAMES, name))
    print(f"  frames in {FRAMES}")


if __name__ == "__main__":
    main()
