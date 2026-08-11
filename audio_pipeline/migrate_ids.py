import json
import os
import argparse

def migrate_ids(metadata_path: str):
    print(f"Lendo metadados de {metadata_path}...")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        
    # Ordenar por filename para ser determinístico
    metadata_list = sorted(metadata_list, key=lambda x: x.get('filename', ''))
    
    new_metadata = []
    for i, meta in enumerate(metadata_list):
        # Apenas atualiza o track_id para o valor numérico (0000, 0001, ...)
        new_meta = meta.copy()
        new_meta['track_id'] = f"{i:04d}"
        new_metadata.append(new_meta)
        
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Sucesso! {len(new_metadata)} registros atualizados.")
    print("Por favor, exclua a pasta chroma_db/ e rode a indexação novamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrar track_ids para numéricos")
    parser.add_argument("--metadata_path", type=str, default="output/metadata.json", help="Caminho para metadata.json")
    
    args = parser.parse_args()
    migrate_ids(args.metadata_path)
