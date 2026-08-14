import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MPD_HOST = "localhost"
MPD_PORT = 6600
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "musicas_embeddings"
MPD_CONF_PATH = os.path.join(BASE_DIR, "mpd.conf")

def get_music_library_root(mpd_conf_path: str) -> str | None:
    """
    Tenta ler a linha music_directory do mpd.conf e retornar o caminho normalizado.
    Se o arquivo não existir ou a linha não for encontrada, retorna None.
    """
    if not os.path.exists(mpd_conf_path):
        return None
        
    try:
        with open(mpd_conf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Procurar pela linha que começa com music_directory e não está comentada
                if line.startswith("music_directory"):
                    parts = line.split(maxsplit=1)
                    if len(parts) > 1:
                        # Remove aspas se houver
                        path = parts[1].strip('"\'')
                        return os.path.normpath(os.path.abspath(path))
    except Exception:
        pass
    
    return None
