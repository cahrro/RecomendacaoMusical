import librosa
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_audio(filepath: str, sr: int = 48000, offset: float = 0.0, duration: float = None) -> np.ndarray:
    """
    Carrega arquivo de áudio e converte para a taxa de amostragem especificada.
    Suporta wav, mp3, flac via librosa e soundfile.
    Retorna o array de áudio e não derruba o pipeline em caso de falha.
    """
    try:
        total_duration = librosa.get_duration(path=filepath)
        if duration is not None and total_duration < offset + duration:
            logger.warning(f"Duração total ({total_duration:.2f}s) menor que offset+duration. Usando fallback para {filepath}.")
            offset = 0.0
            duration = min(45.0, total_duration)
            
        audio, _ = librosa.load(filepath, sr=sr, mono=True, offset=offset, duration=duration)
        return audio
    except Exception as e:
        logger.warning(f"Erro ao carregar áudio {filepath}: {e}")
        return None
