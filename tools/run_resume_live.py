"""Resume a frozen draft on the live instance and report what comes out.

Takes the newest snapshot in the instance's output folder, points the resume
graph at it and runs it, then measures the clip it produced.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/run_resume_live.py [snapshot.json]
"""
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = "http://127.0.0.1:8188"
DRAFTS = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\selflift_drafts"
RESUME = os.path.join(REPO, "workflows", "h3-resume.api.json")


def get(path):
    with urllib.request.urlopen(f"{SERVER}{path}", timeout=60) as handle:
        return json.loads(handle.read().decode("utf-8"))


def post(path, payload):
    request = urllib.request.Request(f"{SERVER}{path}", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{path} -> HTTP {error.code}\n{error.read().decode('utf-8', 'replace')}") from None


def wait_for(prompt_id, timeout=5400):
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = get(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False)[:3000])
            if status.get("completed"):
                return entry
        time.sleep(5)
    raise TimeoutError("the run did not finish")


def main():
    if len(sys.argv) > 1:
        snapshot = sys.argv[1]
    else:
        found = sorted(glob.glob(os.path.join(DRAFTS, "*.json")), key=os.path.getmtime)
        if not found:
            raise SystemExit("no snapshot to resume from")
        snapshot = found[-1]
    print(f"  resuming from {snapshot}")

    with open(RESUME, "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    resume_node = next(key for key, node in graph.items()
                       if node["class_type"] == "SelfLiftH3Resume")
    graph[resume_node]["inputs"]["snapshot_path"] = snapshot

    print("=== posting the resume ===")
    started = time.time()
    prompt_id = post("/prompt", {"prompt": graph})["prompt_id"]
    entry = wait_for(prompt_id)
    print(f"  the resume took {time.time() - started:.0f}s")

    save_node = next(key for key, node in entry["prompt"][2].items()
                     if node["class_type"] == "SaveVideo")
    outputs = entry["outputs"][save_node]
    # SaveVideo reports its file under "images", with the folder as the subfolder.
    clip = (outputs.get("videos") or outputs["images"])[0]
    path = os.path.join(r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output",
                        clip.get("subfolder", ""), clip["filename"])
    print(f"  the clip: {path}")

    import av
    import numpy as np
    with av.open(path) as container:
        kinds = {stream.type for stream in container.streams}
        duration = container.duration / 1_000_000
    with av.open(path) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    with av.open(path) as container:
        chunks = [frame.to_ndarray() for frame in container.decode(audio=0)] if "audio" in kinds else []
    wave = np.concatenate([chunk.reshape(-1) for chunk in chunks]).astype(np.float64) if chunks else None
    print(f"  {len(frames)} frames, {duration:.2f}s, "
          f"size {frames[0].shape[1]}x{frames[0].shape[0]}")
    if wave is not None:
        print(f"  the soundtrack: {len(wave) // 2} samples per channel, "
              f"rms {np.sqrt((wave ** 2).mean()):.5f}, peak {np.abs(wave).max():.4f}")
    else:
        print("  no soundtrack")

    out = r"C:\Users\dream\Desktop\works\comfy-dev\_live_test\frames"
    os.makedirs(out, exist_ok=True)
    from PIL import Image
    for index in (0, len(frames) // 2, len(frames) - 1):
        Image.fromarray(frames[index]).save(os.path.join(out, f"final_{index:03d}.png"))
    print("  frames written to", out)


if __name__ == "__main__":
    main()
