import librosa
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_audio(filepath: str, sr: int = 48000) -> np.ndarray:
    """
    Carrega arquivo de áudio e converte para a taxa de amostragem especificada.
    Suporta wav, mp3, flac via librosa e soundfile.
    Retorna o array de áudio e não derruba o pipeline em caso de falha.
    """
    try:
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        return audio
    except Exception as e:
        logger.warning(f"Erro ao carregar áudio {filepath}: {e}")
        return None
