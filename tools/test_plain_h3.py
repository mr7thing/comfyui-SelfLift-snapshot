"""A plain H3 run with the same prompt, as the control for the draft test.

The draft test's finished clip came out silent. This runs the same prompt
through a plain KSampler -- no SelfLift, no transition -- so the silence can
be told apart from the prompt and from the transition.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_plain_h3.py
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402

from test_live import build_graphs, get, post, say, start_server, wait_for  # noqa: E402

WORK = r"C:\Users\dream\Desktop\works\comfy-dev\_live_test\output"


def rms(path):
    import av

    with av.open(path) as container:
        chunks = [frame.to_ndarray() for frame in container.decode(audio=0)]
    wave = np.concatenate([chunk.reshape(-1) for chunk in chunks]).astype(np.float64)
    return float(np.sqrt((wave ** 2).mean())), float(np.abs(wave).max())


def main():
    draft_graph, _, _ = build_graphs()
    plain = {key: value for key, value in draft_graph.items()
             if key not in ("171", "153", "224")}
    plain["153"] = draft_graph["153"]
    plain["153"]["inputs"]["preview_frames"] = 2
    plain["223"] = draft_graph["223"]          # the plain 8-step schedule
    plain["300"] = {"class_type": "KSampler",
                    "inputs": {"model": ["153", 0], "seed": 684655983592117,
                               "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                               "scheduler": "simple", "denoise": 1.0,
                               "positive": ["136", 0], "negative": ["173", 0],
                               "latent_image": ["136", 1]}}
    plain["301"] = {"class_type": "VAEDecode",
                    "inputs": {"samples": ["300", 0], "vae": ["215", 0]}}
    plain["302"] = {"class_type": "VAEDecodeAudio",
                    "inputs": {"samples": ["300", 0], "vae": ["184", 0]}}
    plain["303"] = {"class_type": "CreateVideo",
                    "inputs": {"images": ["301", 0], "audio": ["302", 0], "fps": 24.0,
                               "bit_depth": 8, "color_space": "sRGB",
                               "chroma_subsampling": "auto", "codec": "none"}}
    plain["304"] = {"class_type": "SaveVideo",
                    "inputs": {"video": ["303", 0], "filename_prefix": "video/plain",
                               "format": "auto", "codec": "auto"}}

    server = start_server()
    try:
        say("=== a plain H3 run, same prompt and seed ===")
        prompt_id = post("/prompt", {"prompt": plain})["prompt_id"]
        wait_for(prompt_id)
        clips = glob.glob(os.path.join(WORK, "video", "plain*.mp4"))
        if not clips:
            raise SystemExit("no clip was written")
        path = max(clips, key=os.path.getmtime)
        level, peak = rms(path)
        say(f"  {os.path.basename(path)}: audio rms {level:.5f}, peak {peak:.4f}")
        if level < 0.01:
            say("  the plain run is silent too: the prompt asks for no sound")
        else:
            say("  the plain run has sound: the silence came from the SelfLift transition")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
