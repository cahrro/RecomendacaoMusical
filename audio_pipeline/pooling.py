import numpy as np

def aggregate_embeddings(embeddings: list[np.ndarray], method: str = "mean") -> np.ndarray:
    """
    Agrega embeddings de múltiplos segmentos em um único vetor global.
    """
    if not embeddings:
        return np.array([])
        
    if method == "mean":
        stacked = np.stack(embeddings)
        return np.mean(stacked, axis=0)
    elif method == "transformer":
        # Hook para implementação futura
        raise NotImplementedError("Transformer pooling ainda não implementado.")
    else:
        raise ValueError(f"Método de pooling '{method}' desconhecido.")
