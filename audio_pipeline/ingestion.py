import librosa
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_audio(filepath: str, sr: int = 48000, offset: float = 0.0, duration: float = None) -> np.ndarray:
    """
    Carrega arquivo de áudio e converte para a taxa de amostragem especificada.
    Suporta wav, mp3, flac via librosa e soundfile.
    Se offset e duration forem passados, carrega apenas o trecho especificado, 
    com fallback seguro para faixas curtas.
    Retorna o array de áudio e não derruba o pipeline em caso de falha.
    """
    try:
        # Pega a duração total do áudio rapidamente sem decodificar tudo
        total_duration = librosa.get_duration(path=filepath)
        
        # Fallback de segurança para faixas muito curtas
        if duration is not None:
            if total_duration < offset + duration:
                logger.info(f"Faixa curta detectada ({total_duration:.1f}s). Ignorando offset inicial para '{filepath}'.")
                offset = 0.0
                duration = min(duration, total_duration)
                
        audio, _ = librosa.load(filepath, sr=sr, mono=True, offset=offset, duration=duration)
        return audio
    except Exception as e:
        logger.warning(f"Erro ao carregar áudio {filepath}: {e}")
        return None
