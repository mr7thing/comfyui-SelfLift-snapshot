"""Pull a few frames out of a clip and write them as PNGs, so a preview can be looked at."""
import os
import sys

import av
import numpy as np
from PIL import Image

clip = sys.argv[1]
out_dir = sys.argv[2]
picks = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 40, 90]
os.makedirs(out_dir, exist_ok=True)

with av.open(clip) as container:
    stream = container.streams.video[0]
    print(f"{clip}: {stream.width}x{stream.height}, {stream.frames} frames, {stream.average_rate}")
    wanted = set(picks)
    for index, frame in enumerate(container.decode(video=0)):
        if index in wanted:
            path = os.path.join(out_dir, f"frame_{index:04d}.png")
            frame.to_image().save(path)
            print("  wrote", path)
