"""Dump the structure of the workflows that matter for the H3 draft/full flow."""

import glob
import json
import os
import sys

INTERESTING = {
    "SelfLiftH3Sampler", "SelfLiftH3Draft", "SelfLiftH3Resume",
    "SaveVideo", "SaveAudio", "VAEDecode", "VAEDecodeAudio", "CreateVideo",
    "easy showAnything", "PreviewImage", "VAELoader",
}


def main(roots):
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*.json"))):
            try:
                data = json.load(open(path, encoding="utf-8"))
            except Exception as error:  # noqa: BLE001
                print("skip", path, error)
                continue
            if "nodes" not in data:
                continue
            print("=" * 24, os.path.basename(path), "nodes", len(data["nodes"]))
            for node in data["nodes"]:
                if node.get("type") not in INTERESTING:
                    continue
                print("  ", node["id"], node["type"], "mode", node.get("mode"))
                print("     in:", [(i.get("name"), i.get("link"), i.get("type"))
                                   for i in node.get("inputs", [])])
                print("     out:", [(o.get("name"), o.get("links"), o.get("type"))
                                    for o in node.get("outputs", [])])
                print("     wv:", node.get("widgets_values"))


if __name__ == "__main__":
    main(sys.argv[1:])
