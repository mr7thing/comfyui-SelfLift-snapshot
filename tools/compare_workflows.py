"""Compare two ComfyUI workflows node by node.

Prints every node of both graphs so the model chain, the attention nodes and the
scheduler can be read side by side.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/compare_workflows.py <a.json> <b.json>
"""
import json
import sys

SKIP = {"Note", "Reroute", "PrimitiveInt"}


def dump(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    print("=" * 30, path.split("\\")[-1].split("/")[-1])
    for node in sorted(data.get("nodes", []), key=lambda item: item["id"]):
        kind = node.get("type")
        if kind in SKIP:
            continue
        print(f"  {node['id']:>4} {kind}")
        widgets = node.get("widgets_values")
        if widgets:
            print(f"        widgets: {widgets}")
        for item in node.get("inputs", []):
            if item.get("link") is not None:
                print(f"        <- {item.get('name')} (link {item.get('link')})")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        dump(argument)
