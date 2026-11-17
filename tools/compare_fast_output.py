"""Is the four-step SelfLift clip actually a clip?

Compares the fast run's output with the twenty-four-step baseline: frame count,
duration, how much detail the frames carry, and how loud the soundtrack is.
Also drops a few frames of each next to each other so they can be looked at.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/compare_fast_output.py
"""
import glob
import os

import numpy as np

FAST = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_test\output\video"
BASE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\video"
FRAMES = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_test\frames"


def newest(folder, prefix):
    found = [os.path.join(folder, name) for name in os.listdir(folder)
             if name.startswith(prefix) and name.endswith(".mp4")]
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def read(path):
    """Frames, audio samples and a detail measure of one clip."""
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
    # A cheap stand-in for detail: how much the pixels change from pixel to pixel.
    detail = float(np.abs(np.diff(pixels, axis=1)).mean())
    samples = np.concatenate([chunk.reshape(-1) for chunk in audio]) if audio else np.zeros(1)
    return frames, float(np.sqrt((samples ** 2).mean())), detail


def main():
    fast = newest(FAST, "fast")
    base = newest(BASE, "ComfyUI_00004")
    if fast is None:
        print("the fast run produced no video")
        return
    for label, path in (("fast  (4 NFE)", fast), ("base (24 NFE)", base)):
        if path is None:
            continue
        frames, rms, detail = read(path)
        print(f"  {label}: {len(frames)} frames, {len(frames) / 24:.2f}s @24fps, "
              f"audio rms {rms:.4f}, detail {detail:.2f}  -> {path}")

    os.makedirs(FRAMES, exist_ok=True)
    from PIL import Image
    for label, path in (("fast", fast), ("base", base)):
        if path is None:
            continue
        frames, _, _ = read(path)
        for index in (0, len(frames) // 2, len(frames) - 1):
            Image.fromarray(frames[index]).save(os.path.join(FRAMES, f"{label}_{index:03d}.png"))
    print(f"  frames in {FRAMES}")


if __name__ == "__main__":
    main()
