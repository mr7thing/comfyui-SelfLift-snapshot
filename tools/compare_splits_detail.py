"""One table for every step split: sharpness, motion and size.

Sharpness is the mean vertical gradient inside a frame -- a rough stand-in for
how much fine detail survived. The 24-step baseline is the yardstick for what
the model can do when it is not being pushed through a four-step schedule.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/compare_splits_detail.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_draft_final import motion, read  # noqa: E402

REFS = r"C:\Users\dream\Desktop\works\comfy-dev\_fast_refs"
NOW = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\video"

CLIPS = [
    ("2 low + 2 high (4 steps)", os.path.join(REFS, "fast_0p4mp_4nfe.mp4")),
    ("3 low + 1 high (4 steps)", os.path.join(NOW, "split_3_1_00001_.mp4")),
    ("4 low + 2 high (6 steps)", os.path.join(NOW, "split_4_2_00001_.mp4")),
    ("5 low + 1 high (6 steps)", os.path.join(NOW, "split_5_1_00001_.mp4")),
    ("24 steps, no split", os.path.join(REFS, "base_0p4mp_24nfe.mp4")),
]


def main():
    print(f"  {'split':<26} {'size':>10} {'sharpness':>11} {'motion':>8}")
    for label, path in CLIPS:
        if not os.path.exists(path):
            print(f"  {label:<26} missing")
            continue
        clip = read(path)
        pixels = clip.astype("float32")
        sharpness = float(abs(pixels[:, 1:] - pixels[:, :-1]).mean())
        print(f"  {label:<26} {f'{clip.shape[2]}x{clip.shape[1]}':>10} {sharpness:11.2f} {motion(clip):8.3f}")


if __name__ == "__main__":
    main()
