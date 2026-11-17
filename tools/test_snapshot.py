"""Check that a frozen draft survives the round trip, and that the draft video draws.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_snapshot.py
"""
import json
import os
import sys
import tempfile

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
REPO = r"C:\Users\dream\Desktop\works\comfy-dev\comfyui-SelfLift-snapshot"
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.dirname(REPO))

# ComfyUI Desktop keeps its models in a shared folder and hands the paths to the
# server on the command line; a standalone run has to pass the same argument.
INSTANCE_PATHS = r"C:\Users\dream\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1790389481797.yaml"
if os.path.exists(INSTANCE_PATHS):
    sys.argv = [sys.argv[0], "--extra-model-paths-config", INSTANCE_PATHS]

import torch  # noqa: E402

import comfy.nested_tensor  # noqa: E402

# ComfyUI Desktop keeps its models in a shared folder and hands the paths to the
# server on the command line; a standalone run loads the same file itself.
INSTANCE_PATHS = r"C:\Users\dream\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1790389481797.yaml"
if os.path.exists(INSTANCE_PATHS):
    import utils.extra_config
    utils.extra_config.load_extra_path_config(INSTANCE_PATHS)


def load(name):
    """Load one of the extension's modules straight from the repository."""
    import importlib.util

    path = os.path.join(REPO, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h3_snapshot = load("h3_snapshot")


def build():
    torch.manual_seed(0)
    video = torch.randn(1, 24, 3, 8, 10)
    audio = torch.randn(1, 32, 2, 5)
    positive = [[
        torch.randn(1, 256, 4096),
        {
            "minimax_refs": [
                {"kind": "image", "latent_h": 4, "latent_w": 5, "latent": torch.randn(1, 24, 1, 4, 5)},
                {"kind": "audio", "ref_audio_t": 7, "audio_latent": torch.randn(1, 32, 2, 7)},
            ],
            "area": (16, 16, 0, 0),
            "mask": torch.zeros(1, 16, 16),
        },
    ]]
    negative = [[torch.randn(1, 256, 4096), {"area": (16, 16, 0, 0), "mask": torch.zeros(1, 16, 16)}]]
    return h3_snapshot.Snapshot(
        state=comfy.nested_tensor.NestedTensor([video, audio]),
        x0=comfy.nested_tensor.NestedTensor([video * 2, audio * 3]),
        nested=True,
        original=comfy.nested_tensor.NestedTensor([torch.randn(1, 24, 3, 16, 20),
                                                   torch.randn(1, 32, 2, 8)]),
        mask=torch.rand(1, 1, 3, 16, 20),
        positive=positive,
        negative=negative,
        sigmas=torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0]),
        seed=4242,
        cfg=5.0,
        params={"transition_step": 2, "lowres_scale": 0.4, "rho": 0.6, "w_min": 0.5, "w_max": 1.0,
                "upscaler_model": "none", "highres_tiling": True, "upscaler_unload": True,
                "latent_extra": {"batch_index": 3}},
        meta={"model": {"class": "MinimaxH3", "latent_format": "MiniMaxH3AV"},
              "prompt": {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "h3.safetensors"}}},
              "ui_workflow": {"nodes": []}},
    )


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if hasattr(left, "tensors"):
        return hasattr(right, "tensors") and same(list(left.tensors), list(right.tensors))
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def check_snapshot():
    print("=== snapshot round trip ===")
    original = build()
    with tempfile.TemporaryDirectory() as folder:
        stem = os.path.join(folder, "draft")
        manifest_path = h3_snapshot.save(original, stem)
        print("  wrote", os.path.basename(manifest_path))
        restored = h3_snapshot.load(manifest_path)

        assert restored.nested == original.nested, "nested flag lost"
        assert restored.seed == original.seed and restored.cfg == original.cfg, "seed/cfg lost"
        assert same(restored.sigmas, original.sigmas), "sigmas changed"
        assert same(restored.params, original.params), "params changed"
        assert same(restored.meta, original.meta), "meta changed"
        for index, (left, right) in enumerate(zip(restored.state.tensors, original.state.tensors)):
            assert same(left, right), f"state stream {index} changed"
        for index, (left, right) in enumerate(zip(restored.x0.tensors, original.x0.tensors)):
            assert same(left, right), f"x0 stream {index} changed"
        assert same(restored.original, original.original), "original latent changed"
        assert same(restored.mask, original.mask), "mask changed"

        meta = restored.positive[0][1]
        assert "area" not in meta and "mask" not in meta, "runtime keys were frozen"
        refs = meta["minimax_refs"]
        assert same(refs[0]["latent"], original.positive[0][1]["minimax_refs"][0]["latent"]), "ref latent changed"
        assert same(refs[1]["audio_latent"], original.positive[0][1]["minimax_refs"][1]["audio_latent"]), "ref audio changed"
        assert refs[0]["latent_h"] == 4 and refs[1]["ref_audio_t"] == 7, "ref metadata changed"
        assert same(restored.positive[0][0], original.positive[0][0]), "text cond changed"

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert manifest["version"] == h3_snapshot.FORMAT_VERSION
        print("  manifest keys:", sorted(manifest.keys()))
        print("  sizes:", {name: round(os.path.getsize(os.path.join(folder, name)) / 1024, 1)
                           for name in sorted(os.listdir(folder))})
    print("  OK")


def check_draft_video():
    print("=== draft video ===")
    h3_draft_video = load("h3_draft_video")
    vae = h3_draft_video.preview_vae()
    assert vae is not None, "no preview VAE found"
    torch.manual_seed(1)
    latent = torch.randn(1, 24, 3, 8, 10)
    frames = h3_draft_video.decode_frames(vae, latent)
    print("  decoded frames:", tuple(frames.shape), "range", float(frames.min()), float(frames.max()))
    assert frames.shape[0] == 5, "H3's five latent tokens should cover five pixel frames here"
    assert frames.shape[3] == 3, "no RGB"
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "draft.mp4")
        h3_draft_video.write_video(frames, path, fps=24.0)
        assert os.path.exists(path) and os.path.getsize(path) > 0, "no video written"
        print("  wrote", os.path.basename(path), os.path.getsize(path), "bytes")
    print("  OK")


check_snapshot()
check_draft_video()
print("\nall checks passed")
