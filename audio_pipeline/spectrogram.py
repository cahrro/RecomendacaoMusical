import numpy as np
import librosa

def compute_mel_spectrogram(segment: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    """
    Calcula o Mel spectrogram do segmento e converte para escala log (dB).
    Usado apenas para inspeção/visualização.
    """
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
