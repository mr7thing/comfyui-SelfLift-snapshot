"""Prove that the refactored sampler still matches the original, bit for bit.

The repository's nodes.py was split into setup / low stage / transition / high
stage so that a draft can stop after the low stage. This test runs the original
implementation (from git) and the new one against the same stubbed sampler and
compares what they produce, first for the full run, then for the draft's frozen
state.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_equivalence.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
REPO = r"C:\Users\dream\Desktop\works\comfy-dev\comfyui-SelfLift-snapshot"
sys.path.insert(0, CORE)

import torch  # noqa: E402

import comfy.latent_formats  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.samplers  # noqa: E402
import latent_preview  # noqa: E402

# ComfyUI Desktop keeps its models in a shared folder and hands the paths to the
# server on the command line; a standalone run loads the same file itself.
INSTANCE_PATHS = r"C:\Users\dream\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1790389481797.yaml"
if os.path.exists(INSTANCE_PATHS):
    import utils.extra_config
    utils.extra_config.load_extra_path_config(INSTANCE_PATHS)

# Every callback the stubbed sampler makes, so the two runs can be compared.
CALLS = []


def fake_sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options=None,
                latent_image=None, callback=None, disable_pbar=False, seed=None):
    """Stand in for ComfyUI's sampler: deterministic, and sensitive to its input."""
    total = len(sigmas) - 1
    x = noise if latent_image is None else noise + latent_image
    for step in range(total):
        x0 = x * 0.5 + 0.125
        CALLS.append({"x0": x0, "x": x, "sigmas": sigmas.clone()})
        if callback is not None:
            callback(step, x0, x, total)
        x = x - (sigmas[step] - sigmas[step + 1]) * 0.05
    return x


comfy.samplers.sample = fake_sample
latent_preview.prepare_callback = lambda model, steps, **kwargs: (lambda step, x0, x, total: None)


class FakeModel:
    """The bits of a ModelPatcher the sampler touches."""

    def __init__(self):
        self.latent_format = comfy.latent_formats.MiniMaxH3AV()
        self.model_sampling = comfy.model_sampling.CONST()
        self.load_device = torch.device("cpu")
        self.model_options = {"sampler_post_cfg_function": []}
        self.model = self

    def get_model_object(self, name):
        return getattr(self, name)

    def process_latent_out(self, latent):
        return latent

    def clone(self):
        return self

    def add_object_patch(self, name, obj):
        setattr(self, name, obj)


class FakeVAE:
    """A stand-in video VAE: 16x spatially, one latent frame per pixel frame."""

    vae_dtype = torch.float32
    device = torch.device("cpu")

    def decode(self, latent):
        batch, channels, frames, height, width = latent.shape
        return torch.full((batch, frames, height * 16, width * 16, 3), 0.25)

    def encode(self, frames):
        if frames.ndim == 4:  # frames-as-batch, the way the pixel anchor calls it
            frames = frames.unsqueeze(0)
        batch, frames, height, width, channels = frames.shape
        return torch.full((batch, 24, frames, height // 16, width // 16), 0.75)


def build_packages(folder):
    """Two importable copies: the current tree, and the tree as git has it."""
    packages = {}
    for name, source in (("selflift_new", REPO), ("selflift_orig", None)):
        target = os.path.join(folder, name)
        os.makedirs(target, exist_ok=True)
        if source is None:
            tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                                     encoding="utf-8", check=True).stdout.split()
            for entry in tracked:
                if entry.endswith(".py"):
                    text = subprocess.run(["git", "show", f"HEAD:{entry}"], cwd=REPO, capture_output=True,
                                          text=True, encoding="utf-8", check=True).stdout
                    with open(os.path.join(target, entry), "w", encoding="utf-8", newline="") as handle:
                        handle.write(text)
        else:
            for entry in os.listdir(source):
                if entry.endswith(".py"):
                    shutil.copy(os.path.join(source, entry), os.path.join(target, entry))
        with open(os.path.join(target, "__init__.py"), "w", encoding="utf-8") as handle:
            handle.write("from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\n")
        packages[name] = target
    sys.path.insert(0, folder)
    return packages


def make_inputs(mask=False, width=64, height=32, frames=3):
    torch.manual_seed(7)
    latent = torch.zeros(1, 32, frames, height // 16, width // 16)
    conditioning = [[torch.randn(1, 256, 4096), {"minimax_refs": [
        {"kind": "image", "latent_h": height // 16, "latent_w": width // 16,
         "latent": torch.randn(1, 24, 1, height // 16, width // 16)}]}]]
    noise_mask = None
    if mask:
        noise_mask = torch.ones(1, 1, frames, height // 16, width // 16)
        noise_mask[:, :, 0] = 0.0
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    return {
        "model": FakeModel(),
        "positive": conditioning,
        "negative": [[torch.randn(1, 256, 4096), {}]],
        "vae": FakeVAE(),
        "latent_image": {"samples": latent, **({"noise_mask": noise_mask} if noise_mask is not None else {})},
        "sampler": comfy.samplers.ksampler("euler"),
        "sigmas": sigmas,
        "seed": 1234,
        "cfg": 5.0,
        "transition_step": 2,
        "lowres_scale": 0.5,
        "rho": 0.6,
        "w_min": 0.5,
        "w_max": 1.0,
    }


def run(module, kwargs, **extra):
    CALLS.clear()
    result = module.progressive_sample(
        kwargs["model"], kwargs["positive"], kwargs["negative"], kwargs["vae"],
        kwargs["latent_image"], kwargs["sampler"], kwargs["sigmas"], kwargs["seed"], kwargs["cfg"],
        kwargs["transition_step"], kwargs["lowres_scale"], kwargs["rho"], kwargs["w_min"],
        kwargs["w_max"], "nearest", **extra)
    return result, list(CALLS)


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(same(left[key], right[key]) for key in left))
    if hasattr(left, "tensors"):
        return same(list(left.tensors), list(right.tensors))
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def check_full_run(orig, new, label, inputs=None, **extra):
    kwargs = make_inputs(**(inputs or {}))
    first, first_calls = run(orig, kwargs, **extra)
    second, second_calls = run(new, kwargs, **extra)
    assert same(first, second), f"{label}: the final latent differs"
    assert len(first_calls) == len(second_calls), f"{label}: different call counts"
    for index, (left, right) in enumerate(zip(first_calls, second_calls)):
        assert same(left["x0"], right["x0"]), f"{label}: call {index} x0 differs"
        assert same(left["x"], right["x"]), f"{label}: call {index} x differs"
    print(f"  {label}: identical ({len(first_calls)} sampler calls)")


def check_draft(orig, new):
    """The draft's frozen state must be the low stage the full run would have used."""
    kwargs = make_inputs()
    _, calls = run(orig, kwargs)
    # The low stage is the first run of sampler calls; its last callback carries
    # the state and the endpoint prediction the transition is built on.
    last_low = calls[kwargs["transition_step"] - 1]
    low_x0, low_state = last_low["x0"], last_low["x"]

    import folder_paths
    folder = tempfile.mkdtemp()
    original_output = folder_paths.get_output_directory
    folder_paths.get_output_directory = lambda: folder
    try:
        draft = new.SelfLiftH3Draft()
        # The node hands back ComfyUI's {"ui", "result"} shape; the values are in "result".
        returned = draft.sample(
            kwargs["model"], kwargs["positive"], kwargs["negative"], kwargs["latent_image"],
            kwargs["sampler"], kwargs["sigmas"], kwargs["seed"], kwargs["cfg"],
            kwargs["transition_step"], kwargs["lowres_scale"], kwargs["rho"], kwargs["w_min"],
            kwargs["w_max"], "none", "test-draft", True, False, True)
    finally:
        folder_paths.get_output_directory = original_output

    assert returned["ui"]["snapshot"][0], "the node reported no snapshot path"
    frozen, manifest_path, video_path = returned["result"]
    assert os.path.exists(manifest_path), "no snapshot written"
    assert video_path and os.path.exists(video_path), "no draft video written"
    snapshot = new.h3_snapshot.load(manifest_path)
    assert same(snapshot.x0.tensors[0], low_x0.tensors[0]), "the draft's endpoint differs from the full run"
    assert same(snapshot.state.tensors[0], low_state.tensors[0]), "the draft's state differs from the full run"
    assert same(snapshot.state.tensors[1], low_state.tensors[1]), "the draft's audio state differs"
    print("  draft: frozen state matches the full run's low stage")
    print("  snapshot:", manifest_path)
    print("  video:", video_path)


def check_resume(orig, new):
    """A draft followed by a resume must land on the full run's result."""
    kwargs = make_inputs()
    expected, _ = run(orig, kwargs)

    import folder_paths
    folder = tempfile.mkdtemp()
    original_output = folder_paths.get_output_directory
    folder_paths.get_output_directory = lambda: folder
    try:
        draft = new.SelfLiftH3Draft()
        returned = draft.sample(
            kwargs["model"], kwargs["positive"], kwargs["negative"], kwargs["latent_image"],
            kwargs["sampler"], kwargs["sigmas"], kwargs["seed"], kwargs["cfg"],
            kwargs["transition_step"], kwargs["lowres_scale"], kwargs["rho"], kwargs["w_min"],
            kwargs["w_max"], "none", "resume-test", False, False, True)
        _, manifest_path, _ = returned["result"]
        resume = new.SelfLiftH3Resume()
        (resumed,) = resume.sample(kwargs["model"], kwargs["vae"], kwargs["sampler"], manifest_path)
    finally:
        folder_paths.get_output_directory = original_output

    assert same(resumed, expected), "the resumed latent differs from the full run"
    print("  resume: lands on the full run's latent")


def main():
    with tempfile.TemporaryDirectory() as folder:
        build_packages(folder)
        import selflift_new.nodes as new
        import selflift_orig.nodes as orig

        print("=== full run against the original ===")
        check_full_run(orig, new, "plain")
        check_full_run(orig, new, "with a noise mask", inputs={"mask": True})

        print("=== draft against the original ===")
        check_draft(orig, new)

        print("=== resume against the original ===")
        check_resume(orig, new)

    print("\nall checks passed")


main()
