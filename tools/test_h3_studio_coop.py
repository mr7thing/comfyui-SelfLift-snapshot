"""Check the H3 Studio cooperation path without running a model.

Three things are checked:

  1. The plain sampler is untouched.  Its keyframe resizing must still be the
     old one, bit for bit -- the cooperation path is a separate switch.
  2. The noise a weakened keyframe carries is really put back.  H3 Studio
     writes a keyframe as `a*z + (1-a)*noise` and keeps `a` as `noise_aug`,
     so the low-resolution keyframe must end up with the same per-pixel noise
     level the full-resolution one had.
  3. The sampler recognises its own TST wrapper, and nobody else's.

Run with any Python that has torch:
  python tools/test_h3_studio_coop.py
"""
import ast
import logging
import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = os.path.join(REPO, "nodes.py")

WANTED = ("_resize_keyframes", "_restore_keyframe_noise", "_keyframe_noise_seed", "_tst_installed",
          "_note_h3_studio_conditions")

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def load_functions(source, names):
    """Pull the helpers out of a nodes.py source without importing ComfyUI."""
    tree = ast.parse(source)
    picked = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in picked}
    if missing:
        raise SystemExit(f"nodes.py has no {', '.join(sorted(missing))}")
    namespace = {"torch": torch, "logging": logging}
    exec(compile(ast.Module(body=picked, type_ignores=[]), NODES, "exec"), namespace)
    return namespace


def legacy_source():
    """The nodes.py as it was committed -- the plain sampler's behaviour."""
    try:
        out = subprocess.run(["git", "-C", REPO, "show", "HEAD:nodes.py"],
                             capture_output=True, encoding="utf-8", errors="replace", check=True)
        return out.stdout
    except Exception as exc:  # pragma: no cover - only when git is missing
        print(f"  (no git history: {exc})")
        return None


def keyframe_cond(latent, noise_aug=None, extra=None):
    kf = {"resolved_frame_index": 0, "latent": latent}
    if noise_aug is not None:
        kf["noise_aug"] = noise_aug
    if extra:
        kf.update(extra)
    return [(torch.zeros(1), {"minimax_keyframes": [kf]})]


def std_of(cond):
    return float(cond[0][1]["minimax_keyframes"][0]["latent"].float().std())


def main():
    with open(NODES, "r", encoding="utf-8") as handle:
        current = handle.read()
    new = load_functions(current, WANTED)

    print("=== 1. the plain sampler's keyframe path is unchanged ===")
    old_source = legacy_source()
    if old_source is None:
        check("legacy comparison available", False, "no git history to compare against")
    else:
        old = load_functions(old_source, ("_resize_keyframes",))
        torch.manual_seed(7)
        latent = torch.randn(1, 16, 1, 64, 64)
        cases = [
            keyframe_cond(latent),
            keyframe_cond(latent, noise_aug=0.649),
            keyframe_cond(torch.randn(1, 16, 3, 32, 48), noise_aug=0.5),
            keyframe_cond(torch.randn(1, 16, 64, 64)),
        ]
        identical = True
        for index, case in enumerate(cases):
            before = old["_resize_keyframes"](case, 20, 24)
            after = new["_resize_keyframes"](case, 20, 24)
            same = torch.equal(before[0][1]["minimax_keyframes"][0]["latent"],
                               after[0][1]["minimax_keyframes"][0]["latent"])
            identical = identical and same
            check(f"case {index}: legacy and current resizing are bit-identical", same)
        check("every case is bit-identical", identical)

    print()
    print("=== 2. the noise of a weakened keyframe is put back ===")
    strength = 0.65
    noise_aug = strength * 0.999
    fraction = 1.0 - noise_aug
    torch.manual_seed(11)
    # a keyframe whose clean part is zero: what is left IS the noise
    noise_only = torch.randn(1, 16, 1, 64, 64) * fraction
    case = keyframe_cond(noise_only, noise_aug=noise_aug)
    legacy = old["_resize_keyframes"](case, 20, 24) if old_source is not None else None
    restored = new["_resize_keyframes"](case, 20, 24, seed=1234, restore_noise=True)
    cells = (64 / 20.0) * (64 / 24.0)
    if legacy is not None:
        legacy_std = std_of(legacy)
        check("the old path really does weaken the noise",
              legacy_std < fraction * 0.9,
              f"old std={legacy_std:.5f}, the full-resolution keyframe carries {fraction:.5f}")
        check("the old path aliases the noise instead of averaging it",
              legacy_std > fraction / cells ** 0.5,
              f"old std={legacy_std:.5f}, a true average would give {fraction / cells ** 0.5:.5f}")
    restored_std = std_of(restored)
    check("the restored keyframe carries the noise level of the full-resolution one",
          abs(restored_std - fraction) / fraction < 0.08,
          f"restored std={restored_std:.5f}, wanted {fraction:.5f}")
    again = new["_resize_keyframes"](case, 20, 24, seed=1234, restore_noise=True)
    check("the restored noise is reproducible",
          torch.equal(restored[0][1]["minimax_keyframes"][0]["latent"],
                      again[0][1]["minimax_keyframes"][0]["latent"]))
    other = new["_resize_keyframes"](case, 20, 24, seed=4321, restore_noise=True)
    check("a different seed gives different noise",
          not torch.equal(restored[0][1]["minimax_keyframes"][0]["latent"],
                          other[0][1]["minimax_keyframes"][0]["latent"]))

    print()
    print("=== 3. a keyframe at full strength is left alone ===")
    torch.manual_seed(13)
    ramp = torch.linspace(0, 1, 64).reshape(1, 1, 1, 1, 64).expand(1, 16, 1, 64, 64).contiguous()
    full_strength = keyframe_cond(ramp.clone(), noise_aug=0.999)
    averaged = new["_resize_keyframes"](full_strength, 20, 24, seed=1, restore_noise=True)
    plain = new["_resize_keyframes"](full_strength, 20, 24)
    want = torch.nn.functional.interpolate(ramp.reshape(1, 16, 64, 64), size=(20, 24), mode="area")
    got = averaged[0][1]["minimax_keyframes"][0]["latent"].reshape(1, 16, 20, 24)
    check("the clean content survives the averaging",
          float((got - want).abs().max()) < 0.02,
          f"max difference {float((got - want).abs().max()):.4f}")
    got_plain = plain[0][1]["minimax_keyframes"][0]["latent"].reshape(1, 16, 20, 24)
    check("a keyframe at full strength gets no noise back",
          float((got - got_plain).abs().max()) < 0.02,
          f"max difference {float((got - got_plain).abs().max()):.4f}")
    no_key = keyframe_cond(torch.randn(1, 16, 1, 64, 64))
    plain_no_key = new["_resize_keyframes"](no_key, 20, 24, seed=1, restore_noise=True)
    # a keyframe that says it carries no noise at all must land on the same numbers
    at_full = keyframe_cond(no_key[0][1]["minimax_keyframes"][0]["latent"], noise_aug=1.0)
    full_no_key = new["_resize_keyframes"](at_full, 20, 24, seed=1, restore_noise=True)
    check("a keyframe without noise_aug gets no noise back",
          torch.equal(plain_no_key[0][1]["minimax_keyframes"][0]["latent"],
                      full_no_key[0][1]["minimax_keyframes"][0]["latent"]))

    print()
    print("=== 4. the sampler recognises its own TST wrapper ===")

    class FakeModel:
        def __init__(self, wrappers):
            self.wrappers = wrappers

    tst_model = FakeModel({"diffusion_model": {"selflift_h3_tst": [lambda: None]}})
    check("the TST wrapper is found", new["_tst_installed"](tst_model) is True)
    other_model = FakeModel({"diffusion_model": {"someone_else": [lambda: None]}})
    check("another attention override is not mistaken for TST",
          new["_tst_installed"](other_model) is False)
    empty_model = FakeModel({})
    check("a model without wrappers is not TST", new["_tst_installed"](empty_model) is False)

    print()
    print("=== 5. the run says out loud what the H3 Studio path changed ===")

    class Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []
            self.records = []

        def emit(self, record):
            self.lines.append(record.getMessage())
            self.records.append(record)

        def warnings(self):
            return [r.getMessage() for r in self.records if r.levelno >= logging.WARNING]

        def clear(self):
            self.lines.clear()
            self.records.clear()

    capture = Capture()
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        new["_note_h3_studio_conditions"](keyframe_cond(torch.zeros(1, 16, 1, 8, 8)),
                                         torch.tensor([0.55, 0.3, 0.0]))
        check("a start below pure noise is called out",
              any("sigma" in line for line in capture.lines),
              capture.lines[0] if capture.lines else "nothing logged")
        capture.clear()
        new["_note_h3_studio_conditions"](keyframe_cond(torch.zeros(1, 16, 3, 8, 8)),
                                         torch.tensor([1.0, 0.5, 0.0]))
        check("a multi-frame keyframe block is called out",
              any("multi-frame" in line for line in capture.lines),
              capture.lines[0] if capture.lines else "nothing logged")
        capture.clear()
        # A plain run -- one keyframe, no label, no strength -- cannot be told
        # apart from a weakened one, so it says so; but it must not warn.
        new["_note_h3_studio_conditions"](keyframe_cond(torch.zeros(1, 16, 1, 8, 8)),
                                         torch.tensor([1.0, 0.5, 0.0]))
        check("a plain run raises no warning", not capture.warnings(), str(capture.lines))
        check("a plain run says what it cannot tell",
              any("cannot tell" in line for line in capture.lines),
              capture.lines[0] if capture.lines else "nothing logged")
        capture.clear()
        # A keyframe at full strength has no noise to put back: nothing to say.
        new["_note_h3_studio_conditions"](
            keyframe_cond(torch.zeros(1, 16, 1, 8, 8), extra={"strength": 1.0}),
            torch.tensor([1.0, 0.5, 0.0]))
        check("a keyframe at full strength says nothing", not capture.lines, str(capture.lines))
        capture.clear()
        # A keyframe that was weakened and lost its label is a real loss: warn.
        new["_note_h3_studio_conditions"](
            keyframe_cond(torch.zeros(1, 16, 1, 8, 8), extra={"strength": 0.65}),
            torch.tensor([1.0, 0.5, 0.0]))
        check("a weakened keyframe without a label is warned about",
              any("weakened" in line for line in capture.warnings()),
              capture.lines[0] if capture.lines else "nothing logged")
        check("the weakened-keyframe warning names the visible effect",
              any("soft" in line for line in capture.warnings()),
              capture.lines[0] if capture.lines else "nothing logged")
        capture.clear()
        # A labelled keyframe proves the per-row mechanism was available, so an
        # unlabelled one next to it can only have been left at full strength.
        both = keyframe_cond(torch.zeros(1, 16, 1, 8, 8), noise_aug=0.649)
        both[0][1]["minimax_keyframes"].append(
            {"resolved_frame_index": 1, "latent": torch.zeros(1, 16, 1, 8, 8)})
        new["_note_h3_studio_conditions"](both, torch.tensor([1.0, 0.5, 0.0]))
        check("an unlabelled keyframe next to a labelled one says nothing",
              not capture.lines, str(capture.lines))
    finally:
        root.setLevel(old_level)
        root.removeHandler(capture)

    print()
    print("FAILURES:", failures if failures else "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
