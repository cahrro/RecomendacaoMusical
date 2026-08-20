import numpy as np

def aggregate_embeddings(embeddings, method: str = "mean") -> np.ndarray:
    """
    Agrega embeddings de múltiplos segmentos em um único vetor global.
    """
    if len(embeddings) == 0:
        return np.array([])
        
    if method == "mean":
        if isinstance(embeddings, list):
            embeddings = np.stack(embeddings)
        return np.mean(embeddings, axis=0)
    elif method == "transformer":
        # Hook para implementação futura
        raise NotImplementedError("Transformer pooling ainda não implementado.")
    else:
        raise ValueError(f"Método de pooling '{method}' desconhecido.")
