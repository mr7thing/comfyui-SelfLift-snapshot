"""Check the draft video at the real H3 shape: 37 latent frames become 124 pixel frames.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_draft_video_scale.py
"""
import os
import sys
import tempfile

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
REPO = r"C:\Users\dream\Desktop\works\comfy-dev\comfyui-SelfLift-snapshot"
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.dirname(REPO))

import torch  # noqa: E402

INSTANCE_PATHS = r"C:\Users\dream\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1790389481797.yaml"
if os.path.exists(INSTANCE_PATHS):
    import utils.extra_config
    utils.extra_config.load_extra_path_config(INSTANCE_PATHS)

import importlib.util  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h3_draft_video = load("h3_draft_video")

# The user's graph: 1344x768 at 0.4 scale -> a 84x48 latent -> a 34x20 low-resolution one.
TARGET = (1344, 768)
LATENT = (1, 24, 37, 20, 34)

print("=== draft video at the real shape ===")
vae = h3_draft_video.preview_vae()
assert vae is not None, "no preview VAE"
torch.manual_seed(3)
latent = torch.randn(*LATENT)
frames = h3_draft_video.decode_frames(vae, latent)
print("  latent", LATENT, "-> frames", tuple(frames.shape))
assert frames.shape[0] == 124, "H3's 37 latent tokens should cover 124 pixel frames"
assert frames.shape[1] == LATENT[3] * 16 and frames.shape[2] == LATENT[4] * 16

frames = h3_draft_video.upscale(frames, TARGET)
print("  upscaled to", tuple(frames.shape))

with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "draft.mp4")
    h3_draft_video.write_video(frames, path, fps=24.0)
    print("  wrote", os.path.getsize(path), "bytes")
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        frames_in_file = stream.frames
        rate = stream.average_rate
    print("  file:", frames_in_file, "frames at", rate)
    assert frames_in_file == 124, f"the file holds {frames_in_file} frames"
    print("  duration", float(frames_in_file) / float(rate), "s (the clip's own length is 124/24 =", 124 / 24, "s)")

print("\nall checks passed")
