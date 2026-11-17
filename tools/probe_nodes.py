import json, sys, os
sys.path.insert(0, r"C:\Users\dream\Desktop\works\comfy-dev\comfyui-SelfLift-snapshot\tools")
import test_fast
graph = test_fast.build_graph("fast")
# keep the prompt out of the file: it is the user's own text
graph["216"]["inputs"]["value"] = "<the prompt from the workflow>"
out = r"C:\Users\dream\Desktop\works\comfy-dev\comfyui-SelfLift-snapshot\workflows\h3-fast.api.json"
with open(out, "w", encoding="utf-8") as handle:
    json.dump(graph, handle, ensure_ascii=False, indent=2)
print("wrote", out, os.path.getsize(out), "bytes")
print("node 171:", json.dumps(graph["171"]["inputs"], ensure_ascii=False))
