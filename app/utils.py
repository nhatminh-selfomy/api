import torchaudio
import torch

def load_audio(filepath: str):
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate

def preprocess_audio(waveform, sample_rate, target_sample_rate=16000):
    if sample_rate != target_sample_rate:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
    return waveform

def audio_to_tensor(filepath: str):
    waveform, sample_rate = load_audio(filepath)
    waveform = preprocess_audio(waveform, sample_rate)
    return waveform
