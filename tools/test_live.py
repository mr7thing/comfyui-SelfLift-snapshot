"""End-to-end proof on the real model: draft + resume equals one full run.

A second ComfyUI is started on its own port with its own output folder, so the
running instance is never touched. Three runs happen there:

  A  the draft graph          -> a snapshot and a draft video
  B  the resume graph         -> the finished clip, resumed from that snapshot
  C  the plain sampler graph  -> the same clip, sampled in one go

B and C must come out of the encoder alike, frame for frame and sample for
sample.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_live.py
"""
import json
import os
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
WORK = os.path.join(os.path.dirname(REPO), "_live_test")
OUTPUT = os.path.join(WORK, "output")


def say(message):
    print(message, flush=True)


def post(path, payload):
    request = urllib.request.Request(f"{SERVER}{path}", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"{path} -> HTTP {error.code}\n{body}") from None


def get(path):
    with urllib.request.urlopen(f"{SERVER}{path}", timeout=60) as handle:
        return json.loads(handle.read().decode("utf-8"))


def start_server():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(OUTPUT, exist_ok=True)
    command = [PYTHON, "-s", "main.py", "--port", str(PORT),
               "--extra-model-paths-config", INSTANCE_PATHS,
               "--input-directory", INPUT_DIR,
               "--output-directory", OUTPUT]
    process = subprocess.Popen(command, cwd=CORE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        time.sleep(2)
        try:
            get("/system_stats")
            say("  server up")
            return process
        except Exception:
            if process.poll() is not None:
                raise RuntimeError("the second ComfyUI died on startup")
    raise RuntimeError("the second ComfyUI did not come up")


def build_graphs():
    with open(os.path.join(REPO, "workflows", "h3-draft.api.json"), "r", encoding="utf-8") as handle:
        draft = json.load(handle)
    with open(os.path.join(REPO, "workflows", "h3-resume.api.json"), "r", encoding="utf-8") as handle:
        resume = json.load(handle)

    # Small and quick: a short clip of 144p, no reference images.
    draft["216"]["inputs"]["value"] = "A single pink petal drifts slowly across a dark studio floor."
    draft["115"]["inputs"]["megapixels"] = 0.1
    draft["132"]["inputs"]["value"] = 0.3
    for node_id in ("332", "333"):
        draft.pop(node_id, None)
    for key in list(draft["136"]["inputs"]):
        if key.startswith("ref_images"):
            draft["136"]["inputs"].pop(key)

    # The full run is the same graph with the plain sampler and a video tail.
    full = json.loads(json.dumps(draft))
    sampler = full["171"]
    sampler["class_type"] = "SelfLiftH3Sampler"
    sampler["inputs"]["vae"] = ["215", 0]
    for key in ("snapshot_name", "draft_video", "audio_vae"):
        sampler["inputs"].pop(key, None)
    full["901"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["171", 0], "vae": ["215", 0]}}
    full["902"] = {"class_type": "VAEDecodeAudio",
                   "inputs": {"samples": ["171", 0], "vae": ["184", 0]}}
    full["903"] = {"class_type": "CreateVideo",
                   "inputs": {"images": ["901", 0], "audio": ["902", 0], "fps": 24.0,
                              "bit_depth": 8, "color_space": "sRGB",
                              "chroma_subsampling": "auto", "codec": "none"}}
    full["904"] = {"class_type": "SaveVideo",
                   "inputs": {"video": ["903", 0], "filename_prefix": "video/full",
                              "format": "auto", "codec": "auto"}}

    resume["153"]["inputs"]["preview_frames"] = 2
    resume["904"]["inputs"]["filename_prefix"] = "video/resumed"
    return draft, resume, full


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
        time.sleep(3)
    raise TimeoutError(f"prompt {prompt_id} did not finish")


def node_of(entry, class_type):
    """The history keys its outputs by the server's own node ids."""
    prompt = entry["prompt"][2]
    for node_id, node in prompt.items():
        if node["class_type"] == class_type:
            return node_id
    raise KeyError(f"no {class_type} in this run")


def newest_video(prefix):
    """The clip SaveVideo wrote last under the given filename prefix."""
    folder = os.path.join(OUTPUT, "video")
    found = [os.path.join(folder, name) for name in os.listdir(folder)
             if name.startswith(prefix) and name.endswith(".mp4")]
    if not found:
        raise KeyError(f"no video with prefix {prefix} in {folder}")
    return max(found, key=os.path.getmtime)


def read_video(path):
    """Every frame and every audio sample of a video, as plain arrays.

    PyAV hands out a container once, so the two streams are read from two
    openings: one decode eats the container.
    """
    import av
    import numpy as np

    with av.open(path) as container:
        kinds = {stream.type for stream in container.streams}
    with av.open(path) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    audio = []
    if "audio" in kinds:
        with av.open(path) as container:
            audio = [frame.to_ndarray() for frame in container.decode(audio=0)]
    return frames, audio


def compare_videos(left_path, right_path):
    """Two clips must be the same clip."""
    import numpy as np

    left_frames, left_audio = read_video(left_path)
    right_frames, right_audio = read_video(right_path)
    assert len(left_frames) == len(right_frames), \
        f"different frame counts: {len(left_frames)} vs {len(right_frames)}"
    for index, (left, right) in enumerate(zip(left_frames, right_frames)):
        assert left.shape == right.shape, f"frame {index}: different shapes"
        if not np.array_equal(left, right):
            difference = np.abs(left.astype(int) - right.astype(int)).max()
            raise AssertionError(f"frame {index} differs, max |difference| = {difference}")
    assert len(left_audio) == len(right_audio), \
        f"different audio chunk counts: {len(left_audio)} vs {len(right_audio)}"
    for index, (left, right) in enumerate(zip(left_audio, right_audio)):
        assert left.shape == right.shape, f"audio {index}: different shapes"
        if not np.array_equal(left, right):
            difference = np.abs(left.astype(float) - right.astype(float)).max()
            raise AssertionError(f"audio {index} differs, max |difference| = {difference}")
    return len(left_frames), sum(chunk.shape[-1] for chunk in left_audio)


def main():
    draft_graph, resume_graph, full_graph = build_graphs()
    server = start_server()
    try:
        info = get("/object_info")
        for name in ("SelfLiftH3Draft", "SelfLiftH3Resume", "SelfLiftH3Sampler"):
            assert name in info, f"{name} is not registered"
        say("  the two new nodes are registered")

        say("=== A: the draft ===")
        started = time.time()
        prompt_id = post("/prompt", {"prompt": draft_graph})["prompt_id"]
        entry = wait_for(prompt_id)
        outputs = entry["outputs"][node_of(entry, "SelfLiftH3Draft")]
        snapshot = outputs["snapshot"][0]
        draft_video = outputs["draft_video"][0]
        say(f"  snapshot: {snapshot}  ({time.time() - started:.0f}s)")
        say(f"  draft video: {draft_video}")
        draft_frames, draft_audio = read_video(draft_video)
        say(f"  the draft has {len(draft_frames)} frames and {len(draft_audio)} audio chunks")

        say("=== B: the resume ===")
        started = time.time()
        resume_graph["900"]["inputs"]["snapshot_path"] = os.path.join(OUTPUT, snapshot)
        prompt_id = post("/prompt", {"prompt": resume_graph})["prompt_id"]
        wait_for(prompt_id)
        resumed = newest_video("resumed")
        say(f"  resumed clip: {os.path.basename(resumed)}  ({time.time() - started:.0f}s)")

        say("=== C: one full run ===")
        started = time.time()
        prompt_id = post("/prompt", {"prompt": full_graph})["prompt_id"]
        wait_for(prompt_id)
        complete = newest_video("full")
        say(f"  full clip: {os.path.basename(complete)}  ({time.time() - started:.0f}s)")

        say("=== the two runs compared ===")
        frames, samples = compare_videos(resumed, complete)
        say(f"  {frames} frames identical")
        say(f"  {samples} audio samples identical")
        say("\nall checks passed")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
