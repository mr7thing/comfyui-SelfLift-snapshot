"""Run the preview the way the canvas runs it, so the saved workflow is a canvas graph.

The API prompt alone leaves the node without a canvas graph to save. This posts
the same prompt with the UI workflow attached as extra data, the way the
frontend does when the run comes from the canvas.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fast import build_graph, get, post  # noqa: E402

SERVER = "http://127.0.0.1:8188"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_WORKFLOW = os.path.join(REPO, "workflows", "H3 SelfLift Draft Mode.json")


def main():
    graph = build_graph("draft")
    with open(UI_WORKFLOW, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    payload = {"prompt": graph,
               "extra_data": {"extra_pnginfo": {"workflow": workflow}}}
    started = time.time()
    prompt_id = post(SERVER, "/prompt", payload)["prompt_id"]
    while time.time() - started < 1800:
        history = get(SERVER, f"/history/{prompt_id}")
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                print(json.dumps(status, ensure_ascii=False)[:2000])
                return 1
            if status.get("completed"):
                break
        time.sleep(3)
    print(f"  the run took {time.time() - started:.0f}s")
    for node_id, output in (history[prompt_id].get("outputs", {}) or {}).items():
        for kind, values in output.items():
            for value in values if isinstance(values, list) else [values]:
                if isinstance(value, str):
                    print(f"  {kind}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
