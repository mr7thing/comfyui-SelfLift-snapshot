"""How much of a sampling step is the live preview worth?

The H3 workflow feeds the same preview frame count to KJ's preview node as it
uses for the clip length, so every sampling step decodes the whole clip with
the tiny preview VAE. That decode happens at the target resolution in both the
low-resolution and the high-resolution stage, which makes it a fixed cost per
step -- and the SelfLift low-resolution callback lifts its latent to the target
grid before handing it over, so it pays the same price.

This measures the decode on its own.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/probe_preview_cost.py
"""
import importlib.util
import os
import sys
import time

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
sys.path.insert(0, CORE)

import torch  # noqa: E402


def load_module(name, path):
    """ComfyUI's own nodes.py shadows the kjnodes package, so load by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(label, function, repeats=2):
    function()  # warm up
    best = None
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
    print(f"  {label}: {best:.2f}s")
    return best


def main():
    import comfy.utils

    tiny_vae = load_module(
        "kj_tiny_vae",
        os.path.join(CORE, "custom_nodes", "comfyui-kjnodes", "nodes", "tiny_vae.py"))

    path = (r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models"
            r"\vae_approx\taeh3.safetensors")
    state = comfy.utils.load_torch_file(path, safe_load=True)
    device = torch.device("cuda")
    if tiny_vae.is_taehv_state_dict(state):
        decoder = tiny_vae.TAEHVDecoder(state, device=device)
    else:
        decoder = tiny_vae.TinyVAEDecoder(state, device=device)
    print(f"  decoder: {type(decoder).__name__} on {device}")

    # What the workflow's preview asks for: every latent frame of the clip.
    indices = list(range(37))
    for label, shape in {
        "low grid  (12,22)": (1, 24, 37, 12, 22),
        "high grid (30,54)": (1, 24, 37, 30, 54),
    }.items():
        latent = torch.zeros(shape, device=device)
        timed(label,
              lambda latent=latent: decoder.decode_video(latent[:1], frame_indices=indices))

    print("\n  the preview the workflow asks for is 124 frames, i.e. the whole clip")


if __name__ == "__main__":
    main()
