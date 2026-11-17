"""Can the SelfLift H3 workflow be made faster without touching any config?

The four-step PDMD run the user just did (545s for the whole clip) differs from
the SelfLift workflow in ways that all live in the graph, not in the install:

  * a distilled four-step model instead of the ref2va base
  * four scheduler steps instead of the twenty-four that
    ExtendIntermediateSigmas stretches the eight-step schedule into
  * reference images sized to the generation ('match') instead of the 2048px
    pipeline ('max'), whose tokens ride along in every step
  * one preview frame per step instead of the whole clip
  * the pytorch attention backend instead of the comfy kitchen one

This runs the graph on a second ComfyUI, so the running instance is never
touched, and prints the per-stage timing the node logs.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_fast.py [fast|base]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
REPO = r"C:\Users\dream\Desktop\works\comfy-dev\comfyui-SelfLift-snapshot"
PYTHON = os.path.join(CORE, ".venv", "Scripts", "python.exe")
INSTANCE_PATHS = r"C:\Users\dream\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1790389481797.yaml"
INPUT_DIR = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input"
PORT = 8189
SERVER = f"http://127.0.0.1:{PORT}"
WORK = os.path.join(os.path.dirname(REPO), "_fast_test")
OUTPUT = os.path.join(WORK, "output")
LOG = os.path.join(WORK, "server.log")
USER_WORKFLOW = (r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
                 r"\user\default\workflows\H3 SelfLift Dual Sample｜Strong Action Ref to Video.json")


def say(message):
    print(message, flush=True)


def post(server, path, payload):
    request = urllib.request.Request(f"{server}{path}", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"{path} -> HTTP {error.code}\n{body}") from None


def get(server, path):
    with urllib.request.urlopen(f"{server}{path}", timeout=60) as handle:
        return json.loads(handle.read().decode("utf-8"))


def start_server():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(OUTPUT, exist_ok=True)
    command = [PYTHON, "-s", "main.py", "--port", str(PORT),
               "--extra-model-paths-config", INSTANCE_PATHS,
               "--input-directory", INPUT_DIR,
               "--output-directory", OUTPUT]
    handle = open(LOG, "w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(command, cwd=CORE, stdout=handle, stderr=subprocess.STDOUT)
    for _ in range(150):
        time.sleep(2)
        try:
            get(SERVER, "/system_stats")
            say("  server up")
            return process
        except Exception:
            if process.poll() is not None:
                raise RuntimeError("the second ComfyUI died on startup")
    raise RuntimeError("the second ComfyUI did not come up")


def build_graph(variant):
    """The user's own prompt and references, with the graph knobs turned."""
    with open(os.path.join(REPO, "workflows", "h3-draft.api.json"), "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    with open(USER_WORKFLOW, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    graph["216"]["inputs"]["value"] = next(
        node["widgets_values"][0] for node in workflow["nodes"] if node["type"] == "PrimitiveStringMultiline")
    images = [node["widgets_values"][0] for node in workflow["nodes"] if node["type"] == "Image Load"]
    for node_id, name in (("332", images[0]), ("333", images[1])):
        graph[node_id]["inputs"]["image"] = name.split(" [")[0]

    # The full sampler: the draft node is only the low-resolution prefix.
    sampler = graph["171"]
    sampler["class_type"] = "SelfLiftH3Sampler"
    sampler["inputs"]["vae"] = ["215", 0]
    sampler["inputs"].pop("audio_vae", None)
    sampler["inputs"].pop("snapshot_name", None)
    sampler["inputs"].pop("draft_video", None)

    # The video tail: the sampler feeds the decoder the way the user's graph does.
    graph["901"] = {"class_type": "VAEDecode",
                    "inputs": {"samples": ["171", 0], "vae": ["215", 0]}}
    graph["902"] = {"class_type": "VAEDecodeAudio",
                    "inputs": {"samples": ["171", 0], "vae": ["184", 0]}}
    graph["903"] = {"class_type": "CreateVideo",
                    "inputs": {"images": ["901", 0], "audio": ["902", 0], "fps": 25.0,
                               "bit_depth": 8, "color_space": "sRGB",
                               "chroma_subsampling": "auto", "codec": "none"}}
    graph["904"] = {"class_type": "SaveVideo",
                    "inputs": {"video": ["903", 0], "filename_prefix": f"video/{variant}",
                               "format": "auto", "codec": "auto"}}

    if variant in ("fast", "preview"):
        # 1. the distilled four-step model
        graph["254"]["inputs"]["unet_name"] = "PRISM_DMAD_PDMD_w6a8_g32.safetensors"
        graph.pop("255", None)                       # the turbo LoRA belongs to the base model
        graph["258"]["inputs"]["model"] = ["254", 0]
        # 2. four scheduler steps: drop the node that stretched eight into twenty-four
        graph["223"]["inputs"]["steps"] = 4
        graph.pop("224", None)
        sampler["inputs"]["sigmas"] = ["223", 0]
        sampler["inputs"]["transition_step"] = 2
        # 3. references sized to the generation instead of the 2048px pipeline
        graph["136"]["inputs"]["ref_image_size"] = "match"
        # 4. one preview frame instead of the whole clip
        graph["153"]["inputs"]["preview_frames"] = 1
        # 5. the sparse attention the four-step workflow uses
        graph["328"]["inputs"]["selection"] = "vsa"
        graph["328"]["inputs"].pop("selection.tau", None)
        graph["328"]["inputs"]["selection.keep_percent"] = 10
        # 6. the pytorch attention backend
        graph["256"]["inputs"]["attention"] = "pytorch attention"

    if variant == "preview":
        # The whole run on a quarter of the canvas: 0.16 MP instead of 0.4 MP.
        graph["115"]["inputs"]["megapixels"] = 0.16
        sampler["inputs"]["lowres_scale"] = 0.5
    return graph


def wait_for(prompt_id, timeout=7200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = get(SERVER, f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False)[:3000])
            if status.get("completed"):
                return entry
        time.sleep(3)
    raise TimeoutError("the run did not finish")


def report_timing():
    """What the node logged about the run."""
    with open(LOG, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    for line in text.splitlines():
        if "SelfLift timing" in line or "SelfLift plan" in line:
            print("   ", line.split("] ", 1)[-1] if "] " in line else line)


def report_environment():
    """The build the server started on, and which quantized backend it picked."""
    with open(LOG, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    print("  the build:")
    for line in text.splitlines():
        if ("pytorch version" in line or "PYTORCH CUDA VERSION" in line
                or "Found comfy_kitchen backend cuda" in line):
            cleaned = line.split("] ", 1)[-1] if "] " in line else line
            print("   ", cleaned[:220])


def newest_video():
    folder = os.path.join(OUTPUT, "video")
    found = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".mp4")]
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "fast"
    graph = build_graph(variant)
    server = start_server()
    try:
        info = get(SERVER, "/object_info")
        assert "SelfLiftH3Sampler" in info, "the sampler node is not registered"
        say(f"=== the {variant} graph ===")
        started = time.time()
        try:
            prompt_id = post(SERVER, "/prompt", {"prompt": graph})["prompt_id"]
        except RuntimeError as error:
            say(f"  the prompt was refused:\n{error}")
            report_timing()
            return 1
        entry = wait_for(prompt_id)
        elapsed = time.time() - started
        say(f"\n  the run took {elapsed:.0f}s")
        video = newest_video()
        if video:
            say(f"  clip: {video} ({os.path.getsize(video) / 2**20:.1f} MiB)")
        say("  the node's own timings:")
        report_timing()
        report_environment()
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
