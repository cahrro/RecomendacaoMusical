import argparse
import json
import logging
import os
from tqdm import tqdm

from fingerprint import generate_fingerprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def backfill(metadata_path: str, music_root: str):
    if not os.path.exists(metadata_path):
        logger.error(f"Arquivo {metadata_path} não encontrado.")
        return
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        
    changed = 0
    errors = 0
    
    logger.info(f"Analisando {len(metadata_list)} faixas em {metadata_path}...")
    
    for entry in tqdm(metadata_list, desc="Gerando fingerprints"):
        existing_fp = entry.get("fingerprint")
        # Ignora apenas se já possuir fingerprint RAW (com vírgulas)
        if existing_fp and ',' in existing_fp:
            continue

            
        rel_filepath = entry.get("filepath")
        if not rel_filepath:
            continue
            
        abs_filepath = os.path.join(music_root, rel_filepath)
        if not os.path.exists(abs_filepath):
            logger.warning(f"Arquivo não encontrado: {abs_filepath}")
            errors += 1
            continue
            
        fp = generate_fingerprint(abs_filepath)
        if fp:
            entry["fingerprint"] = fp
            changed += 1
        else:
            errors += 1
            
    if changed > 0:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Sucesso! {changed} fingerprints adicionados e salvos em {metadata_path}.")
    else:
        logger.info("Nenhuma modificação feita no arquivo de metadados.")
        
    if errors > 0:
        logger.warning(f"Ocorreram {errors} erros (arquivos ausentes ou falha na geração).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera fingerprints acústicos para faixas existentes no metadata.json")
    parser.add_argument("--metadata", type=str, required=True, help="Caminho do metadata.json")
    parser.add_argument("--music_root", type=str, required=True, help="Caminho absoluto da raiz da biblioteca de músicas")
    
    args = parser.parse_args()
    backfill(args.metadata, args.music_root)
