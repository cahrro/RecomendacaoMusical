import os
import json
import logging
import numpy as np

from fingerprint import generate_fingerprint, find_duplicate_by_fingerprint
from pipeline import process_track
import vector_store

logger = logging.getLogger(__name__)

def load_known_fingerprints(metadata_path: str) -> dict[str, str]:
    """
    Carrega {track_id: fingerprint} de todas as entradas já indexadas no metadata.json.
    """
    if not os.path.exists(metadata_path):
        return {}
        
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
            
        known_fps = {}
        for entry in metadata_list:
            track_id = str(entry.get("track_id"))
            fp = entry.get("fingerprint")
            if track_id and fp:
                known_fps[track_id] = fp
        return known_fps
    except Exception as e:
        logger.error(f"Erro ao carregar fingerprints do metadata.json: {e}")
        return {}

def index_unknown_track(filepath: str, metadata_path: str, collection, known_fingerprints: dict, music_library_root: str) -> str | None:
    """
    Fluxo de indexação sob demanda (lazy indexing).
    Verifica duplicatas via fingerprint acústico. Se for nova, roda pipeline de extração e insere no banco.
    """
    logger.info(f"Iniciando indexação sob demanda para: {filepath}")
    
    # 1. Gerar fingerprint
    new_fp = generate_fingerprint(filepath)
    if not new_fp:
        logger.error(f"Falha ao gerar fingerprint para {filepath}. Indexação abortada.")
        return None
        
    # 2. Verificar duplicata
    duplicate_track_id = find_duplicate_by_fingerprint(new_fp, known_fingerprints, threshold=0.95)
    
    # Carregar metadata
    metadata_list = []
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
            
    norm_filepath = os.path.normpath(filepath)
    norm_root = os.path.normpath(music_library_root)
    
    if os.path.isabs(norm_filepath):
        if norm_filepath.lower().startswith(norm_root.lower()):
            rel_filepath = os.path.relpath(norm_filepath, norm_root)
        else:
            rel_filepath = filepath
    else:
        abs_path = os.path.join(norm_root, norm_filepath)
        rel_filepath = os.path.relpath(abs_path, norm_root)
        
    rel_filepath = rel_filepath.replace('\\', '/')
    filename = os.path.basename(filepath)
    
    if duplicate_track_id:
        logger.info(f"Faixa '{filename}' identificada como duplicata de track_id={duplicate_track_id}")
        
        # Adicionar novo filepath aos aliases
        for entry in metadata_list:
            if str(entry.get("track_id")) == duplicate_track_id:
                if "aliases" not in entry:
                    entry["aliases"] = []
                if rel_filepath not in entry["aliases"] and entry.get("filepath") != rel_filepath:
                    entry["aliases"].append(rel_filepath)
                break
                
        # Salvar metadata atualizado
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            
        return duplicate_track_id
        
    # 3. Não é duplicata, gerar novo embedding e inserir
    
    # Inferir subgenero a partir da pasta
    parent_dir = os.path.dirname(rel_filepath)
    subgenero = os.path.basename(parent_dir) if parent_dir else "desconhecido"
    if not subgenero:
        subgenero = "desconhecido"
        
    # Gerar novo track_id sequencial
    max_id = -1
    for entry in metadata_list:
        try:
            tid = int(entry.get("track_id", -1))
            if tid > max_id:
                max_id = tid
        except ValueError:
            pass
            
    new_id_int = max_id + 1 if max_id >= 0 else 0
    new_track_id = f"{new_id_int:04d}"
    
    # Rodar pipeline
    result = process_track(filepath, track_id=new_track_id, subgenero=subgenero, rel_filepath=rel_filepath)
    
    if not result:
        logger.error(f"Falha ao processar a faixa {filepath} no pipeline.")
        return None
        
    embedding = result.pop("embedding") # Remover do dict
    result["fingerprint"] = new_fp # Adicionar o fingerprint
    
    # Inserir no ChromaDB usando upsert
    try:
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        collection.upsert(
            ids=[new_track_id],
            embeddings=[embedding],
            metadatas=[{"filename": filename, "subgenero": subgenero}]
        )
    except Exception as e:
        logger.error(f"Erro ao inserir {new_track_id} no ChromaDB: {e}")
        return None
        
    # Atualizar metadata_list e salvar
    metadata_list.append(result)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Nova faixa indexada: track_id={new_track_id}, subgenero={subgenero}")
    
    # Atualizar o known_fingerprints em memória
    known_fingerprints[new_track_id] = new_fp
    
    return new_track_id
