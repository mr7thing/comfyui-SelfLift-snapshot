"""Where does the written mp4 lose its frames?"""
import os
import sys
import tempfile

REPO = r"C:\Users\dream\Desktop\works\comfy-dev\comfyui-SelfLift-snapshot"
sys.path.insert(0, REPO)
sys.path.insert(0, r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI")

import torch  # noqa: E402
import av  # noqa: E402
from fractions import Fraction  # noqa: E402


def write(path, with_audio):
    frames = torch.rand(12, 64, 96, 3)
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=Fraction(24, 1))
    stream.width = 96
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    astream = None
    if with_audio:
        astream = container.add_stream("aac", rate=32000, layout="stereo")
        astream.bit_rate = 128000
    for frame in frames:
        array = torch.clamp(frame * 255.0, 0.0, 255.0)
        container.mux(stream.encode(av.VideoFrame.from_ndarray(
            array.to(torch.uint8).cpu().numpy(), format="rgb24")))
    container.mux(stream.encode())
    if astream is not None:
        wave = torch.zeros(2, 32000)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=32000)
        frame = av.AudioFrame.from_ndarray(wave.numpy(), format="fltp", layout="stereo")
        frame.sample_rate = 32000
        parts = list(resampler.resample(frame)) + list(resampler.resample(None))
        written = 0
        for part in parts:
            part.pts = written
            part.time_base = Fraction(1, 32000)
            container.mux(astream.encode(part))
            written += part.samples
        container.mux(astream.encode())
    container.close()


def report(path):
    print(f"  {os.path.basename(path)}: {os.path.getsize(path)} bytes")
    with av.open(path) as container:
        print("    streams:", [(s.type, s.codec_context.name, s.frames, s.duration) for s in container.streams])
        try:
            frames = list(container.decode(video=0))
            print(f"    container.decode(video=0): {len(frames)} frames")
        except Exception as error:
            print(f"    container.decode(video=0) failed: {error}")
    with av.open(path) as container:
        try:
            frames = list(container.decode(audio=0))
            print(f"    container.decode(audio=0): {len(frames)} frames")
        except Exception as error:
            print(f"    container.decode(audio=0) failed: {error}")
    with av.open(path) as container:
        try:
            packets = list(container.demux())
            print("    packets:", len(packets))
        except Exception as error:
            print("    demux failed", error)


with tempfile.TemporaryDirectory() as folder:
    for name, with_audio in (("video.mp4", False), ("both.mp4", True)):
        path = os.path.join(folder, name)
        try:
            write(path, with_audio)
        except Exception as error:
            print(f"  writing {name} failed: {type(error).__name__}: {error}")
            continue
        report(path)
