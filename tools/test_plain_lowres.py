"""A plain H3 run on the prefix's own grid, as the control for the draft.

The draft samples the low-resolution prefix of a 0.4 MP run, so the grid is
416x224 (the prefix of 864x480 at lowres_scale 26/54). This runs the same
prompt, the same seed and the same references through a plain KSampler on that
same grid, with no SelfLift and no transition: the same noise, the same
schedule, the whole thing at low resolution.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_plain_lowres.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fast import build_graph, get, post  # noqa: E402

SERVER = "http://127.0.0.1:8188"
OUTPUT = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output"


def main():
    graph = build_graph("fast")
    # The plain sampler replaces the SelfLift node, on the prefix's own grid.
    graph.pop("171")
    graph["115"]["inputs"]["megapixels"] = 0.1          # 416x224
    graph["136"]["inputs"]["ref_image_size"] = "max"    # the same reference tokens as the draft
    graph["300"] = {"class_type": "KSampler",
                    "inputs": {"model": ["153", 0], "seed": 684655983592117,
                               "steps": 4, "cfg": 1.0, "sampler_name": "euler",
                               "scheduler": "simple", "denoise": 1.0,
                               "positive": ["136", 0], "negative": ["173", 0],
                               "latent_image": ["136", 1]}}
    graph["901"]["inputs"]["samples"] = ["300", 0]
    graph["902"]["inputs"]["samples"] = ["300", 0]
    graph["904"]["inputs"]["filename_prefix"] = "video/plainlow"

    started = time.time()
    prompt_id = post(SERVER, "/prompt", {"prompt": graph})["prompt_id"]
    deadline = time.time() + 3600
    while time.time() < deadline:
        history = get(SERVER, f"/history/{prompt_id}")
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                print(status)
                return 1
            if status.get("completed"):
                break
        time.sleep(3)
    print(f"  the plain low-resolution run took {time.time() - started:.0f}s")
    folder = os.path.join(OUTPUT, "video")
    clips = [os.path.join(folder, name) for name in os.listdir(folder) if name.startswith("plainlow")]
    if clips:
        clip = max(clips, key=os.path.getmtime)
        print(f"  clip: {clip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
