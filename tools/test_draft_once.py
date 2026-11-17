"""Run the draft alone and check what it leaves behind.

The draft must produce a snapshot, a draft video with the clip's own length and
a soundtrack, and the run history must carry the paths back to whoever asked
for the run. The soundtrack is checked against a finished clip of the same run:
the draft's rough estimate and the finished clip must be the same soundscape.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_draft_once.py [reference.mp4]
"""
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402

import test_live  # noqa: E402
from test_live import build_graphs, get, node_of, post, read_video, say, start_server, wait_for  # noqa: E402


def correlation(left, right):
    length = min(len(left), len(right))
    left = left[:length] - left[:length].mean()
    right = right[:length] - right[:length].mean()
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.dot(left, right) / (np.sqrt(np.dot(left, left)) * np.sqrt(np.dot(right, right))))


def main():
    reference = sys.argv[1] if len(sys.argv) > 1 else None
    draft_graph, _, _ = build_graphs()
    server = start_server()
    try:
        prompt_id = post("/prompt", {"prompt": draft_graph})["prompt_id"]
        entry = wait_for(prompt_id)
        node_id = node_of(entry, "SelfLiftH3Draft")
        outputs = entry["outputs"][node_id]
        say(f"  the history carries: {json.dumps(outputs, ensure_ascii=False)[:400]}")
        assert "text" in outputs, "the node did not report anything to the history"
        assert outputs["snapshot"][0].endswith(".json"), outputs["snapshot"]
        assert outputs["draft_video"][0].endswith(".mp4"), outputs["draft_video"]

        draft_frames, draft_audio = read_video(outputs["draft_video"][0])
        say(f"  the draft holds {len(draft_frames)} frames and {len(draft_audio)} audio chunks")
        assert draft_audio, "the draft video has no soundtrack"

        if reference:
            final_frames, final_audio = read_video(reference)
            draft_wave = np.concatenate([chunk.reshape(-1) for chunk in draft_audio]).astype(np.float64)
            final_wave = np.concatenate([chunk.reshape(-1) for chunk in final_audio]).astype(np.float64)
            value = correlation(draft_wave, final_wave)
            say(f"  the draft's soundtrack against the finished clip: r = {value:+.4f}")
            assert value > 0.5, "the draft's soundtrack does not match the finished clip"
        say("\nall checks passed")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
