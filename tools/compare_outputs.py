"""Compare the clips a live run left behind.

The end-to-end run writes three files: the draft video, the clip resumed from
the snapshot and the clip sampled in one go. This checks them without loading
a model: the two finished clips must be the same clip, and the draft's
soundtrack must be the same soundscape the finished clip ends up with.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/compare_outputs.py
"""
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402

from test_live import OUTPUT, compare_videos, read_video  # noqa: E402


def newest(pattern):
    found = glob.glob(os.path.join(OUTPUT, pattern))
    if not found:
        raise KeyError(f"nothing matches {pattern} under {OUTPUT}")
    return max(found, key=os.path.getmtime)


def main():
    resumed = newest("video/resumed*.mp4")
    complete = newest("video/full*.mp4")
    drafts = sorted(glob.glob(os.path.join(OUTPUT, "selflift_drafts", "*.mp4")))
    if not drafts:
        raise KeyError("no draft video")
    draft = drafts[-1]
    print(f"  draft   {os.path.basename(draft)}")
    print(f"  resumed {os.path.basename(resumed)}")
    print(f"  full    {os.path.basename(complete)}")

    frames, samples = compare_videos(resumed, complete)
    print(f"  the two finished clips match: {frames} frames, {samples} audio samples")

    draft_frames, draft_audio = read_video(draft)
    final_frames, final_audio = read_video(resumed)
    print(f"  the draft holds {len(draft_frames)} frames and {len(draft_audio)} audio chunks")
    print(f"  the final clip holds {len(final_frames)} frames and {len(final_audio)} audio chunks")

    if not draft_audio or not final_audio:
        raise AssertionError("one of the clips has no soundtrack")
    draft_wave = np.concatenate([chunk.reshape(-1) for chunk in draft_audio])
    final_wave = np.concatenate([chunk.reshape(-1) for chunk in final_audio])
    length = min(len(draft_wave), len(final_wave))
    draft_wave = draft_wave[:length].astype(np.float64)
    final_wave = final_wave[:length].astype(np.float64)
    correlation = np.corrcoef(draft_wave, final_wave)[0, 1]
    print(f"  the draft's soundtrack correlates with the finished one: r = {correlation:.4f}")
    assert correlation > 0.5, "the draft's soundtrack does not match the finished clip"
    print("\nall checks passed")


if __name__ == "__main__":
    main()
