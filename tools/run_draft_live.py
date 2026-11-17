"""Run the draft on the live instance with the user's own prompt and references.

Takes the prompt text and the image names straight out of the user's workflow
so the draft is the real thing, not a stand-in.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/run_draft_live.py
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
WORKFLOW = (r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
            r"\user\default\workflows\H3 SelfLift Dual Sample｜Strong Action Ref to Video.json")
DRAFT = os.path.join(REPO, "workflows", "h3-draft.api.json")


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


def wait_for(prompt_id, timeout=3600):
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
    with open(WORKFLOW, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    prompt_text = next(node["widgets_values"][0] for node in workflow["nodes"]
                       if node["type"] == "PrimitiveStringMultiline")
    images = [node["widgets_values"][0] for node in workflow["nodes"]
              if node["type"] == "Image Load"]
    print(f"  the prompt is {len(prompt_text)} characters, the references are {images}")

    with open(DRAFT, "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    graph["216"]["inputs"]["value"] = prompt_text
    # The Image Load node lists the input folder by name and the output folder
    # with a folder prefix, so a file that lives in the output root is named
    # without one.
    for node_id, name in (("332", images[0]), ("333", images[1])):
        graph[node_id]["inputs"]["image"] = name.split(" [")[0]

    print("=== posting the draft ===")
    started = time.time()
    prompt_id = post("/prompt", {"prompt": graph})["prompt_id"]
    entry = wait_for(prompt_id)
    print(f"  the run took {time.time() - started:.0f}s")

    node_id = next(key for key, node in entry["prompt"][2].items()
                   if node["class_type"] == "SelfLiftH3Draft")
    outputs = entry["outputs"][node_id]
    print(f"  snapshot: {outputs['snapshot'][0]}")
    print(f"  draft video: {outputs['draft_video'][0]}")

    import av
    import numpy as np
    path = outputs["draft_video"][0]
    with av.open(path) as container:
        kinds = {stream.type for stream in container.streams}
    with av.open(path) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    with av.open(path) as container:
        chunks = [frame.to_ndarray() for frame in container.decode(audio=0)] if "audio" in kinds else []
    pixels = np.stack([frame.mean(axis=(1, 2)) for frame in frames])
    print(f"  the draft holds {len(frames)} frames, brightness {pixels.mean():.1f} "
          f"(a black draft would sit near 0), and {len(chunks)} audio chunks")
    print(f"  every 20th frame's brightness: {[round(value, 1) for value in pixels[::20]]}")


if __name__ == "__main__":
    main()
