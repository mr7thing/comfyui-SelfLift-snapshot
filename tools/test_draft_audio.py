"""The draft's mp4 must carry both the picture and the soundtrack.

No model is loaded here: a stand-in audio VAE mimics the real one's output
shape, and the written file is read back with PyAV.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/test_draft_audio.py
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
sys.path.insert(0, REPO)
sys.path.insert(0, CORE)

import torch  # noqa: E402

import h3_draft_video  # noqa: E402


class FakeAudioVAE:
    """The real H3 audio VAE hands back [B, samples, channels]."""

    audio_sample_rate = 32000

    def __init__(self, samples=32000, channels=2):
        self.samples = samples
        self.channels = channels

    def decode(self, latent):
        batch, _, stereo, frames = latent.shape
        length = frames * 800
        wave = torch.zeros(batch, length, stereo)
        for channel in range(stereo):
            time_axis = torch.arange(length, dtype=torch.float32)
            wave[:, :, channel] = torch.sin(time_axis * (0.01 + 0.005 * channel))
        return wave


def main():
    frames = torch.rand(12, 64, 96, 3)
    audio_vae = FakeAudioVAE()
    latent = torch.zeros(1, 32, 2, 40)

    waveform, rate = h3_draft_video.decode_audio(audio_vae, latent)
    assert waveform.shape == (1, 2, 32000), waveform.shape
    assert rate == 32000, rate
    print(f"  waveform {tuple(waveform.shape)} at {rate} Hz")

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "draft.mp4")
        h3_draft_video.write_video(frames, path, fps=24.0,
                                   audio=waveform[0], audio_sample_rate=rate)
        assert os.path.exists(path)

        import av
        with av.open(path) as container:
            kinds = {stream.type: stream for stream in container.streams}
            assert "video" in kinds, list(kinds)
            assert "audio" in kinds, list(kinds)
        with av.open(path) as container:
            video_frames = list(container.decode(video=0))
        with av.open(path) as container:
            audio_chunks = list(container.decode(audio=0))
            samples = sum(chunk.samples for chunk in audio_chunks)
            print(f"  the file holds {len(video_frames)} frames and {samples} audio samples")
            assert len(video_frames) == 12, len(video_frames)
            assert abs(samples - 32000) <= 2048, samples
            print(f"  video h264, audio aac at "
                  f"{[s.codec_context.sample_rate for s in container.streams if s.type == 'audio'][0]} Hz")

        # Without an audio VAE the file is a silent mp4, as before.
        quiet = os.path.join(folder, "quiet.mp4")
        h3_draft_video.write_video(frames, quiet, fps=24.0)
        with av.open(quiet) as container:
            kinds = {stream.type for stream in container.streams}
            assert kinds == {"video"}, kinds
        print("  the silent draft still works")
    print("all checks passed")


if __name__ == "__main__":
    main()
