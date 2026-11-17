"""How close is the draft's soundtrack to the finished clip's?

Raw waveforms of a rough estimate and a finished track rarely line up; the
energy over time is the fairer comparison. This measures both.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/check_draft_sound.py
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

AUDIO_VAE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI\..\..\..\ComfyUI-Shared\models\vae\minimax_h3_audio_vae_fp32.safetensors"
AUDIO_VAE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\vae\minimax_h3_audio_vae_fp32.safetensors"
RATE = 32000
WINDOW = int(RATE * 0.05)


def clip_audio(path):
    import av

    with av.open(path) as container:
        chunks = [frame.to_ndarray() for frame in container.decode(audio=0)]
    return np.concatenate([chunk.reshape(-1) for chunk in chunks]).astype(np.float64)


def envelope(wave):
    length = (len(wave) // WINDOW) * WINDOW
    blocks = wave[:length].reshape(-1, WINDOW)
    return np.sqrt((blocks ** 2).mean(axis=1))


def correlate(left, right):
    length = min(len(left), len(right))
    left = left[:length] - left[:length].mean()
    right = right[:length] - right[:length].mean()
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.dot(left, right) / (np.sqrt(np.dot(left, left)) * np.sqrt(np.dot(right, right))))


def main():
    manifest = glob.glob(os.path.join(OUTPUT, "selflift_drafts", "*.json"))[0]
    tensors = comfy.utils.load_torch_file(manifest.replace(".json", ".safetensors"), safe_load=True)
    audio_latent = tensors["x0.1"].float()

    final = clip_audio(sorted(glob.glob(os.path.join(OUTPUT, "video", "resumed*.mp4")))[-1])
    print(f"  the finished clip: {len(final)} samples, {len(envelope(final))} envelope steps")

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(AUDIO_VAE, safe_load=True))
    wave = vae.decode(audio_latent).detach().movedim(-1, 1)
    std = torch.std(wave, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    wave = (wave / std).clamp(-1.0, 1.0)[0].reshape(-1).numpy().astype(np.float64)
    print(f"  the frozen endpoint decodes to {len(wave)} samples")
    print(f"  raw waveform correlation:  r = {correlate(wave, final):+.4f}")
    print(f"  envelope correlation:     r = {correlate(envelope(wave), envelope(final)):+.4f}")
    print(f"  the draft's rms {np.sqrt((wave ** 2).mean()):.4f}, "
          f"the finished one {np.sqrt((final ** 2).mean()):.4f}")


if __name__ == "__main__":
    main()
