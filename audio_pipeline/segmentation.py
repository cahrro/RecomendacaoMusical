import numpy as np

def segment_audio(audio: np.ndarray, sr: int, window_sec: float = 10.0, hop_sec: float = 5.0) -> list[np.ndarray]:
    """
    Segmenta o áudio em janelas com sobreposição.
    Descartar segmentos residuais menores que 2 segundos.
    """
    window_length = int(window_sec * sr)
    hop_length = int(hop_sec * sr)
    min_length = int(2.0 * sr)
    
    segments = []
    total_length = len(audio)
    
    for start in range(0, total_length, hop_length):
        end = start + window_length
        segment = audio[start:end]
        
        # Descartar se for menor que 2s
        if len(segment) >= min_length:
            segments.append(segment)
            
    return segments
