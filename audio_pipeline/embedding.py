import logging
import torch
import numpy as np
from transformers import ClapModel, ClapProcessor

logger = logging.getLogger(__name__)

# Padrão Singleton para modelo e processor
_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        logger.info(f"Carregando modelo CLAP em dispositivo: {_DEVICE}")
        model_name = "laion/clap-htsat-unfused"
        _PROCESSOR = ClapProcessor.from_pretrained(model_name)
        _MODEL = ClapModel.from_pretrained(model_name).to(_DEVICE)
        _MODEL.eval()

def generate_embedding(audio_segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Gera embedding para um segmento de áudio usando o modelo CLAP.
    O áudio bruto é passado.
    """
    _load_model()
    
    # Verificar a duração e avisar se o CLAP pode estar cortando (o padrão costuma ser 10s ou algo próximo para HTSAT)
    duration_sec = len(audio_segment) / sr
    if duration_sec > 10.0:
        logger.debug(f"Aviso: Segmento tem {duration_sec:.2f}s, pode ser truncado pelo processador CLAP.")
        
    with torch.no_grad():
        inputs = _PROCESSOR(audio=audio_segment, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        
        outputs = _MODEL.get_audio_features(**inputs)
        embedding = outputs.pooler_output.cpu().numpy()
        
    # Espera-se que seja shape (1, hidden_size)
    return embedding[0]

def generate_embeddings_batch(segments_list: list[np.ndarray], sr: int) -> np.ndarray:
    """
    Gera embeddings para uma lista de segmentos de áudio em lote usando o modelo CLAP.
    Otimiza muito o processamento na CPU paralelizando as operações matriciais.
    """
    if not segments_list:
        return np.array([])
        
    _load_model()
    
    with torch.no_grad():
        inputs = _PROCESSOR(audio=segments_list, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        
        outputs = _MODEL.get_audio_features(**inputs)
        embeddings = outputs.pooler_output.cpu().numpy()
        
    # Retorna shape (n_segments, hidden_size)
    return embeddings
