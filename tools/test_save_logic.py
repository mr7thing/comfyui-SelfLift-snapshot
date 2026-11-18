"""Check the run record without running a model.

The decisions -- who fed the save node, whether that was a preview, and where
the files land -- are plain functions, so they can be checked on their own. The
workflow check runs against the graph this repository ships.

Run with any Python:
  python tools/test_save_logic.py
"""
import ast
import copy
import json
import os
import sys
import tempfile
import time
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = os.path.join(REPO, "nodes.py")
WORKFLOW = os.path.join(REPO, "workflows", "H3 SelfLift Draft Mode.json")

WANTED = ("_upstream_sampler", "_runs_in_draft", "_switch_off_draft", "_draft_widget_slot", "_file_away")


def load_functions():
    """Pull the helpers out of nodes.py without importing ComfyUI."""
    with open(NODES, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    picked = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in WANTED]
    missing = set(WANTED) - {node.name for node in picked}
    if missing:
        raise SystemExit(f"nodes.py has no {', '.join(sorted(missing))}")
    namespace = {"os": os, "json": json, "copy": copy, "time": time, "uuid": uuid}
    exec(compile(ast.Module(body=picked, type_ignores=[]), NODES, "exec"), namespace)
    return namespace


def main():
    helpers = load_functions()
    upstream = helpers["_upstream_sampler"]
    preview = helpers["_runs_in_draft"]
    switch_off = helpers["_switch_off_draft"]
    file_away = helpers["_file_away"]
    problems = []

    # The sampler that fed the save node, read from the save node's own input.
    prompt = {
        "171": {"class_type": "SelfLiftH3DraftSampler", "inputs": {"draft_mode": True}},
        "905": {"class_type": "SelfLiftH3Save", "inputs": {"latent": ["171", 0]}},
        "901": {"class_type": "VAEDecode", "inputs": {"samples": ["905", 0]}},
    }
    if upstream(prompt, "905") != "171":
        problems.append("the sampler that fed the save node was not found")
    if not preview(prompt, "171"):
        problems.append("a draft run was not recognised as one")
    if preview(prompt, "905"):
        problems.append("the save node was mistaken for a sampler")

    # The switch off is the run's own.
    off = {"171": {"class_type": "SelfLiftH3DraftSampler", "inputs": {"draft_mode": False}}}
    if preview(off, "171"):
        problems.append("a full run was mistaken for a draft")

    # A sampler that is not the draft sampler is always a full run.
    other = {"171": {"class_type": "SelfLiftH3Sampler", "inputs": {}}}
    if preview(other, "171"):
        problems.append("the plain sampler was mistaken for a draft")

    # The saved workflow: the same graph with the switch off.
    with open(WORKFLOW, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    switch_off(workflow, "171")
    node = next(item for item in workflow["nodes"] if item["id"] == 171)
    values = node["widgets_values"]
    if node["widgets_values_named"]["draft_mode"] is not False:
        problems.append("the named widget value was not switched off")
    if values[9] is not False:
        problems.append(f"the widget value was not switched off: {values}")
    if values[8] != "minimax_h3_latent_upscaler_3d_fp16_V2.safetensors":
        problems.append("another widget was switched off by mistake")
    if values[10] != "taeh3.safetensors" or values[13] is not True:
        problems.append("the widgets after the switch were moved")

    # Filing away: a file moves, a text is written out.
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "staged.mp4")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write("video")
        target = os.path.join(folder, "filed.mp4")
        file_away(source, target)
        if os.path.exists(source) or not os.path.exists(target):
            problems.append("a file was not moved into place")
        text_target = os.path.join(folder, "filed.json")
        file_away('{"a": 1}', text_target)
        if not os.path.exists(text_target):
            problems.append("a text was not written out")

    for problem in problems:
        print("  PROBLEM:", problem)
    print("  ok" if not problems else f"  {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
