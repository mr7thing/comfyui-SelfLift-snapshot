"""Is the draft's soundtrack the soundscape the finished clip has?

The sampler carries H3's audio stream scaled onto the video schedule (4x for
the default shifts), so a frozen endpoint has to be brought back before the
audio VAE can read it. This decodes the snapshot's audio both ways and checks
which one lines up with the finished clip's soundtrack.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/check_draft_audio.py
"""
import glob
import os
import sys

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

from test_live import OUTPUT  # noqa: E402

AUDIO_VAE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\vae\minimax_h3_audio_vae_fp32.safetensors"


def clip_audio(path):
    import av

    with av.open(path) as container:
        chunks = [frame.to_ndarray() for frame in container.decode(audio=0)]
    return np.concatenate([chunk.reshape(-1) for chunk in chunks]).astype(np.float64)


def decode(latent, vae):
    audio = vae.decode(latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = (audio / std).clamp(-1.0, 1.0)
    return audio[0].detach().reshape(-1).numpy().astype(np.float64)


def correlation(left, right):
    length = min(len(left), len(right))
    left = left[:length] - left[:length].mean()
    right = right[:length] - right[:length].mean()
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.dot(left, right) / (np.sqrt(np.dot(left, left)) * np.sqrt(np.dot(right, right))))


def main():
    manifest = glob.glob(os.path.join(OUTPUT, "selflift_drafts", "*.json"))
    if not manifest:
        raise SystemExit("no snapshot to read")
    tensors = comfy.utils.load_torch_file(manifest[0].replace(".json", ".safetensors"), safe_load=True)
    audio_latent = tensors["x0.1"].float()
    print(f"  the snapshot holds an audio latent of {tuple(audio_latent.shape)}")

    final = clip_audio(sorted(glob.glob(os.path.join(OUTPUT, "video", "resumed*.mp4")))[-1])
    print(f"  the finished clip's soundtrack: {len(final)} samples")

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(AUDIO_VAE, safe_load=True))
    for scale in (4.0, 1.0):
        wave = decode(audio_latent / scale, vae)
        print(f"  decoded at 1/{scale:g}: r = {correlation(wave, final):+.4f}  "
              f"(rms {np.sqrt((wave ** 2).mean()):.4f})")


if __name__ == "__main__":
    main()
