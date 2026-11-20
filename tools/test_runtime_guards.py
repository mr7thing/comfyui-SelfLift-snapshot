"""Check the runtime guards that protect the H3 Studio cooperation path.

These are the checks that turn a silent miscalculation into a message:

  1. B -- the packed sequence must end with the video segment.  TST locates the
     video rows by counting backwards from the end, so a trailing segment makes
     it correct the wrong rows without raising anything.
  2. A -- the core keeps one attention-rewrite slot.  A second writer silently
     replaces the first, so installing into an occupied slot has to be refused.
  3. C -- a multi-frame keyframe block is downscaled with the low-resolution
     prefix, which softens the join it was meant to make seamless.
  4. D -- a keyframe without a noise_aug label cannot have its noise put back,
     so its weakening is unknown.

Run with any Python that has torch:
  python tools/test_runtime_guards.py
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TST = os.path.join(REPO, "h3_tst.py")
NODES = os.path.join(REPO, "nodes.py")

failures = []


def check(name, ok, detail=""):
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + detail) if detail else ""))
    if not ok:
        failures.append(name)


def load_functions(path, names):
    """Pull the helpers out of a source file without importing ComfyUI."""
    tree = ast.parse(open(path, "r", encoding="utf-8").read())
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in picked}
    if missing:
        raise SystemExit("%s has no %s" % (os.path.basename(path), ", ".join(sorted(missing))))
    return {n.name: n for n in picked}


class FakeLayout:
    """What the core puts in transformer_options under minimax_h3_layout."""

    def __init__(self, segments):
        self.segments = segments


def make_state(frames, rows):
    return {"frames": frames, "rows_per_frame": rows,
            "layout_checked": False, "layout_error": None}


def main():
    funcs = load_functions(TST, ["_check_packed_order", "_check_override_slot"])
    scope = {}
    exec(compile(ast.Module(body=[funcs["_check_packed_order"]], type_ignores=[]),
                 "<check>", "exec"), scope)
    check_order = scope["_check_packed_order"]

    frames, rows = 24, 100
    video_rows = frames * rows

    normal = FakeLayout([(0, 40, "text"), (40, 240, "cond"), (240, video_rows + 240, "video")])
    state = make_state(frames, rows)
    check_order(state, {"minimax_h3_layout": normal})
    check("B: the normal layout passes the segment-order check",
          state["layout_checked"] and state["layout_error"] is None)

    trailing = FakeLayout([(0, 40, "text"), (40, 240, "cond"),
                           (240, video_rows + 240, "video"),
                           (video_rows + 240, video_rows + 280, "ref_img")])
    state = make_state(frames, rows)
    raised = None
    try:
        check_order(state, {"minimax_h3_layout": trailing})
    except ValueError as error:
        raised = str(error)
    check("B: a segment after the video raises", raised is not None)
    if raised:
        for word in ("video", "wrong rows", "minimax_h3_layout"):
            check("B: the message names %r" % word, word in raised)

    # a video segment that is the last one but holds the wrong number of rows:
    # the packed sequence would be short of rows, and TST would read past the video
    short = FakeLayout([(0, 40, "text"), (40, video_rows + 10, "video")])
    state = make_state(frames, rows)
    raised = None
    try:
        check_order(state, {"minimax_h3_layout": short})
    except ValueError as error:
        raised = str(error)
    check("B: a video segment with the wrong row count raises", raised is not None)

    long_video = FakeLayout([(0, 40, "text"), (40, video_rows + 80, "video")])
    state = make_state(frames, rows)
    raised = None
    try:
        check_order(state, {"minimax_h3_layout": long_video})
    except ValueError as error:
        raised = str(error)
    check("B: a video segment with too many rows raises", raised is not None)

    state = make_state(frames, rows)
    check_order(state, {})
    check("B: no layout in the options does not raise",
          state["layout_error"] is None and not state["layout_checked"])

    # the error must be raised on every forward, not only the first one:
    # a dedup guard would let a caught exception through on the next forward
    state = make_state(frames, rows)
    raised_first = raised_second = None
    try:
        check_order(state, {"minimax_h3_layout": trailing})
    except ValueError as error:
        raised_first = str(error)
    try:
        check_order(state, {"minimax_h3_layout": trailing})
    except ValueError as error:
        raised_second = str(error)
    check("B: the error is raised again on the next forward", raised_second is not None)
    check("B: the repeated error is the same message", raised_first == raised_second)

    scope2 = {}
    exec(compile(ast.Module(body=[funcs["_check_override_slot"]], type_ignores=[]),
                 "<check>", "exec"), scope2)
    check_slot = scope2["_check_override_slot"]

    class FakeModel:
        def __init__(self, options):
            self.model_options = options

    check("A: an empty override slot is not reported",
          check_slot(FakeModel({"transformer_options": {}})) is None)

    def other_override(*args, **kwargs):
        return None

    occupied = FakeModel({"transformer_options": {"optimized_attention_override": other_override}})
    check("A: an occupied override slot is reported", check_slot(occupied) is not None)

    node_src = open(NODES, "r", encoding="utf-8").read()
    node_funcs = load_functions(NODES, ["_note_h3_studio_conditions"])
    # ast.dump renders logging.warning as Attribute(value=Name(id='logging'), attr='warning'),
    # so look for the call node itself instead of the source text
    warning_calls = [
        node for node in ast.walk(node_funcs["_note_h3_studio_conditions"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "warning"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logging"
    ]
    check("C: the multi-frame case is raised as a warning", len(warning_calls) >= 2)
    check("C: the warnings are not errors", not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("error", "exception")
        for node in ast.walk(node_funcs["_note_h3_studio_conditions"])))
    check("C: the warning names the visible effect", "join" in node_src and "softer" in node_src)
    check("D: the warning names the missing label", "noise_aug" in node_src and "unlabelled" in node_src)

    print("")
    if failures:
        print("%d check(s) failed: %s" % (len(failures), ", ".join(failures)))
        return 1
    print("all runtime-guard checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
