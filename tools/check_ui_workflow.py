"""Check a UI workflow for dangling links and mismatched input slots."""
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    workflow = json.load(handle)

nodes = {node["id"]: node for node in workflow["nodes"]}
problems = []
seen = set()

for link in workflow["links"]:
    link_id, origin_id, origin_slot, target_id, target_slot, kind = link
    seen.add(link_id)
    origin = nodes.get(origin_id)
    target = nodes.get(target_id)
    if origin is None or target is None:
        problems.append(f"link {link_id} points at a missing node")
        continue
    outputs = origin.get("outputs") or []
    if origin_slot >= len(outputs):
        problems.append(f"link {link_id} uses output slot {origin_slot} of node {origin_id}")
    elif link_id not in (outputs[origin_slot].get("links") or []):
        problems.append(f"link {link_id} is not listed on node {origin_id} output {origin_slot}")
    inputs = target.get("inputs") or []
    if target_slot >= len(inputs):
        problems.append(f"link {link_id} uses input slot {target_slot} of node {target_id}")
    elif inputs[target_slot].get("link") != link_id:
        problems.append(f"link {link_id} is not set on node {target_id} input {target_slot}")

for node in workflow["nodes"]:
    for index, item in enumerate(node.get("inputs") or []):
        link_id = item.get("link")
        if link_id is not None and link_id not in seen:
            problems.append(f"node {node['id']} input {index} ({item['name']}) points at link {link_id}, which is gone")
    for index, item in enumerate(node.get("outputs") or []):
        for link_id in item.get("links") or []:
            if link_id not in seen:
                problems.append(f"node {node['id']} output {index} lists link {link_id}, which is gone")

print(f"{path}: {len(workflow['nodes'])} nodes, {len(workflow['links'])} links")
for problem in problems:
    print("  PROBLEM:", problem)
print("  ok" if not problems else f"  {len(problems)} problems")
