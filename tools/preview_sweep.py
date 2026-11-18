"""How large can the draft preview be, and what does each option cost?

The draft preview is drawn from the low-resolution prefix, so its pixel count is
`target megapixels * lowres_scale`. That gives two different ways to make the
preview sharper, and they cost very different things:

  * raise the target: the preview grows with it, but so does the finished clip
    (the high-resolution steps and the lift both scale with the target area)
  * raise lowres_scale: the preview grows, the finished clip does not -- but the
    prefix is then a different grid, so the draft only predicts the full run if
    the full run uses the same lowres_scale

This measures both: the preview's own resolution and the seconds it costs, and
(for the full runs) what the finished clip costs at that target.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/preview_sweep.py draft:0.4:0.4 full:0.4:0.6

Each spec is mode:target_megapixels:lowres_scale[:transition_step].
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fast import (LOG, OUTPUT, SERVER, build_graph, get, post, start_server,  # noqa: E402
                       wait_for)

DRAFTS = os.path.join(OUTPUT, "selflift")


def video_size(path):
    """The pixel size of an mp4, without pulling in a decoder stack here."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        return stream.codec_context.width, stream.codec_context.height


def build(spec):
    """The graph for one sweep entry."""
    mode, megapixels, scale, steps = spec
    if mode == "draft":
        graph = build_graph("draft")
    else:
        graph = build_graph("fast")
    sampler = graph["171"]
    graph["115"]["inputs"]["megapixels"] = float(megapixels)
    sampler["inputs"]["lowres_scale"] = float(scale)
    if steps is not None:
        sampler["inputs"]["transition_step"] = int(steps)
    if mode == "draft":
        sampler["inputs"]["preview_video"] = True
    return graph


def log_tail(marker):
    """The node's own timing lines since the last marker."""
    with open(LOG, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    lines = [line.split("] ", 1)[-1] for line in text.splitlines()
             if "SelfLift timing" in line or "SelfLift draft" in line]
    return lines


def newest(folder, suffix=".mp4"):
    if not os.path.isdir(folder):
        return None
    found = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(suffix)]
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def run(spec):
    """Run one entry and report what it cost and what it produced."""
    mode, megapixels, scale, steps = spec
    graph = build(spec)
    before_drafts = set(os.listdir(DRAFTS)) if os.path.isdir(DRAFTS) else set()
    before_videos = set(os.listdir(os.path.join(OUTPUT, "video"))) \
        if os.path.isdir(os.path.join(OUTPUT, "video")) else set()

    started = time.time()
    prompt_id = post(SERVER, "/prompt", {"prompt": graph})["prompt_id"]
    wait_for(prompt_id)
    elapsed = time.time() - started

    preview = None
    if os.path.isdir(DRAFTS):
        fresh = [n for n in os.listdir(DRAFTS) if n not in before_drafts and n.endswith(".mp4")]
        if fresh:
            preview = os.path.join(DRAFTS, max(fresh, key=lambda n: os.path.getmtime(os.path.join(DRAFTS, n))))
    clip = None
    video_folder = os.path.join(OUTPUT, "video")
    if os.path.isdir(video_folder):
        fresh = [n for n in os.listdir(video_folder) if n not in before_videos and n.endswith(".mp4")]
        if fresh:
            clip = os.path.join(video_folder, max(fresh, key=lambda n: os.path.getmtime(os.path.join(video_folder, n))))

    entry = {"spec": spec, "seconds": round(elapsed, 1),
             "preview": preview, "clip": clip}
    if preview:
        entry["preview_size"] = video_size(preview)
    if clip:
        entry["clip_size"] = video_size(clip)
    entry["timings"] = [line for line in log_tail(None)[-14:]]
    return entry


def main():
    specs = []
    for raw in sys.argv[1:]:
        parts = raw.split(":")
        mode = parts[0]
        megapixels = float(parts[1])
        scale = float(parts[2])
        steps = int(parts[3]) if len(parts) > 3 and parts[3] else None
        specs.append((mode, megapixels, scale, steps))

    server = start_server()
    results = []
    try:
        info = get(SERVER, "/object_info")
        assert "SelfLiftH3Draft" in info, "the draft node is not registered"
        for spec in specs:
            print(f"\n=== {spec} ===", flush=True)
            try:
                entry = run(spec)
            except Exception as error:
                print(f"  failed: {error}")
                results.append({"spec": spec, "error": str(error)[:400]})
                continue
            results.append(entry)
            print(f"  {entry['seconds']}s")
            if entry.get("preview"):
                print(f"  preview {entry['preview_size'][0]}x{entry['preview_size'][1]}  {entry['preview']}")
            if entry.get("clip"):
                print(f"  clip    {entry['clip_size'][0]}x{entry['clip_size'][1]}  {entry['clip']}")
            for line in entry["timings"]:
                print("   ", line[:200])
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()

    with open(os.path.join(OUTPUT, "preview_sweep.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print(f"\n  summary -> {os.path.join(OUTPUT, 'preview_sweep.json')}")
    print(f"\n  {'spec':<28}{'s':>8}  preview          clip")
    for entry in results:
        if "error" in entry:
            print(f"  {str(entry['spec']):<28}{'FAILED':>8}  {entry['error'][:60]}")
            continue
        preview = (f"{entry['preview_size'][0]}x{entry['preview_size'][1]}" if entry.get("preview") else "-")
        clip = (f"{entry['clip_size'][0]}x{entry['clip_size'][1]}" if entry.get("clip") else "-")
        print(f"  {str(entry['spec']):<28}{entry['seconds']:>8}  {preview:<17} {clip}")


if __name__ == "__main__":
    main()
