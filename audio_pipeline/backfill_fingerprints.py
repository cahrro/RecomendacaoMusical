import os
import json
import logging
from fingerprint import generate_fingerprint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backfill(metadata_path: str, music_root: str):
    """
    Percorre o metadata.json existente e gera os fingerprints faltantes para cada arquivo.
    """
    if not os.path.exists(metadata_path):
        logger.error(f"Arquivo metadata não encontrado em: {metadata_path}")
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        
    updated = False
    
    for entry in metadata_list:
        filepath = entry['filepath']
        abs_path = filepath
        if not os.path.isabs(filepath):
            abs_path = os.path.join(music_root, filepath)
            
        if not os.path.exists(abs_path):
            logger.warning(f"Arquivo não encontrado para backfill: {abs_path}")
            continue
            
        logger.info(f"Gerando fingerprint (raw) para: {abs_path}")
        fp = generate_fingerprint(abs_path)
        
        if fp:
            entry['fingerprint'] = fp
            updated = True
        else:
            logger.error(f"Falha ao gerar fingerprint para: {abs_path}")
            
    if updated:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        logger.info("Backfill concluído com sucesso e metadata.json atualizado.")
    else:
        logger.info("Nenhum fingerprint novo foi gerado (todos já existem ou ocorreram erros).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="./results/metadata.json")
    parser.add_argument("--music_root", type=str, default="C:/mpd/music")
    args = parser.parse_args()
    
    backfill(args.metadata, args.music_root)
