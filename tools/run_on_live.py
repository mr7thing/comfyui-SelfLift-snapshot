"""Post one of the test graphs to the ComfyUI that is already running.

Unlike tools/test_fast.py this does not start a second instance: it uses the
one the user is looking at, so the run lands in the shared output folder.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/run_on_live.py [fast|draft|preview|base]
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fast import build_graph  # noqa: E402

SERVER = "http://127.0.0.1:8188"
OUTPUT = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output"


def post(path, payload):
    request = urllib.request.Request(f"{SERVER}{path}", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"{path} -> HTTP {error.code}\n{body}") from None


def get(path):
    with urllib.request.urlopen(f"{SERVER}{path}", timeout=60) as handle:
        return json.loads(handle.read().decode("utf-8"))


def newest_video(folder):
    if not os.path.isdir(folder):
        return None
    found = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".mp4")]
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "draft"
    graph = build_graph(variant)
    before = newest_video(os.path.join(OUTPUT, "video"))
    started = time.time()
    try:
        prompt_id = post("/prompt", {"prompt": graph})["prompt_id"]
    except RuntimeError as error:
        print(f"  the prompt was refused:\n{error}")
        return 1
    deadline = time.time() + 7200
    while time.time() < deadline:
        history = get(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                print(json.dumps(status, ensure_ascii=False)[:3000])
                return 1
            if status.get("completed"):
                break
        time.sleep(3)
    print(f"  the {variant} run took {time.time() - started:.0f}s")
    clip = newest_video(os.path.join(OUTPUT, "video"))
    if clip and clip != before:
        print(f"  clip: {clip} ({os.path.getsize(clip) / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
