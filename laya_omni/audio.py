"""Reading audio files, without torch (the teacher-labelling client runs where torch is not installed)."""

import numpy as np


def load_audio(path, sampling_rate=16000):
    """Any file librosa can read, as mono float32 at the encoder's sampling rate."""
    import librosa

    try:
        wav, _ = librosa.load(path, sr=sampling_rate, mono=True)
    except Exception:  # formats libsndfile cannot read (AAC / m4a, some mp3s): decode with ffmpeg via PyAV
        wav = _decode_av(path, sampling_rate)
    return wav.astype(np.float32)


def _decode_av(path, sampling_rate):
    import av

    chunks = []
    with av.open(str(path)) as f:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sampling_rate)
        for packet in f.demux(audio=0):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:  # a corrupt packet: skip it, keep the rest of the clip
                continue
            chunks += [r.to_ndarray().reshape(-1) for frame in frames for r in resampler.resample(frame)]
        chunks += [r.to_ndarray().reshape(-1) for r in resampler.resample(None)]
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
