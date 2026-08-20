import os
import glob
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from time import time

from ingestion import load_audio
from segmentation import segment_audio
from embedding import generate_embedding, generate_embeddings_batch
from pooling import aggregate_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_track(filepath: str, track_id: str, subgenero: str, rel_filepath: str) -> dict:
    """
    Processa uma única faixa: carrega, segmenta, extrai embeddings e agrega.
    """
    start_time = time()
    sr = 48000
    
    # 1. Ingestão (Amostragem Inteligente: pula 15s, lê 45s)
    audio = load_audio(filepath, sr=sr, offset=15.0, duration=45.0)
    if audio is None:
        return None
        
    # 2. Segmentação (janelas de 10s, hop de 5s)
    segments = segment_audio(audio, sr=sr, window_sec=10.0, hop_sec=5.0)
    if not segments:
        logger.warning(f"Faixa {filepath} não possui segmentos válidos (muito curta?).")
        return None
        
    # 3. Extração de Embeddings (Em Lote / Batch Inference)
    if len(segments) > 0:
        segment_embeddings = generate_embeddings_batch(segments, sr=sr)
    else:
        logger.warning(f"Faixa {filepath} não gerou segmentos.")
        return None
        
    # 4. Pooling
    global_embedding = aggregate_embeddings(segment_embeddings, method="mean")
    
    elapsed = time() - start_time
    logger.info(f"Processado {filepath} em {elapsed:.2f}s ({len(segments)} segmentos).")
    
    basename = os.path.basename(filepath)
    
    return {
        "track_id": track_id,
        "filename": basename,
        "filepath": rel_filepath,
        "subgenero": subgenero,
        "n_segments": len(segments),
        "embedding": global_embedding
    }

def process_directory(folder: str, output_dir: str, **kwargs):
    """
    Processa um diretório de arquivos de áudio e salva embeddings em npz e metadados em json.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    from pathlib import Path
    folder_path = Path(folder).resolve()
    
    # Encontrar arquivos suportados usando os.walk
    extensions = {".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC"}
    filepaths_by_subgenre = {}
    for root, dirs, files in os.walk(folder_path):
        subgenero = Path(root).name
        if subgenero == folder_path.name:
            continue # ignora arquivos na raiz
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                if subgenero not in filepaths_by_subgenre:
                    filepaths_by_subgenre[subgenero] = []
                filepaths_by_subgenre[subgenero].append(Path(root) / file)
                
    # Aplicar limite e unificar a lista
    limit_per_genre = kwargs.get("limit_per_genre", None)
    filepaths = []
    for sg, paths in filepaths_by_subgenre.items():
        paths = sorted(paths)
        if limit_per_genre:
            paths = paths[:limit_per_genre]
        filepaths.extend(paths)
        
    filepaths = sorted(list(set(filepaths)), key=lambda p: (p.parent.name, p.name))
    
    if not filepaths:
        logger.warning(f"Nenhum arquivo de áudio encontrado em {folder}")
        return
        
    logger.info(f"Iniciando processamento de {len(filepaths)} arquivos.")
    
    all_embeddings = []
    all_metadata = []
    subgenre_counts = {}
    
    for i, p in enumerate(tqdm(filepaths, desc="Processando faixas")):
        try:
            filepath = str(p)
            subgenero = p.parent.name
            
            try:
                rel_filepath = str(p.relative_to(folder_path))
            except ValueError:
                rel_filepath = p.name
                
            track_id_str = f"{i:04d}"
            result = process_track(filepath, track_id=track_id_str, subgenero=subgenero, rel_filepath=rel_filepath)
            if result is not None:
                all_embeddings.append(result["embedding"])
                
                # Remover embedding do dict de metadados antes de salvar no JSON
                meta = {k: v for k, v in result.items() if k != "embedding"}
                all_metadata.append(meta)
                
                subgenre_counts[subgenero] = subgenre_counts.get(subgenero, 0) + 1
        except Exception as e:
            logger.error(f"Erro inesperado ao processar {filepath}: {e}")
            continue
            
    # Relatório final por subgênero
    logger.info("\n=== Resumo por Subgênero ===")
    if not subgenre_counts:
        logger.warning("Nenhum arquivo processado com sucesso.")
    for sg, count in subgenre_counts.items():
        logger.info(f"{sg}: {count} faixas")
    logger.info("============================\n")
        
    if all_embeddings:
        # Salvar arrays de embeddings
        embeddings_array = np.stack(all_embeddings)
        out_npz = os.path.join(output_dir, "embeddings.npz")
        np.savez_compressed(out_npz, embeddings=embeddings_array)
        
        # Salvar metadados
        out_json = os.path.join(output_dir, "metadata.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Salvos {len(all_embeddings)} vetores em {out_npz}")
        logger.info(f"Salvos metadados em {out_json}")
    else:
        logger.warning("Nenhum embedding gerado com sucesso. Nenhum arquivo salvo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Extração de Embeddings de Áudio")
    parser.add_argument("--input", type=str, help="Pasta de entrada contendo áudios")
    parser.add_argument("--output", type=str, help="Pasta de saída para embeddings.npz e metadata.json")
    parser.add_argument("--test-file", type=str, help="Caminho para um arquivo único para testar o pipeline rapidamente")
    
    parser.add_argument("--limit", type=int, help="Limite de músicas por subgênero (ideal para testes rápidos)")
    
    args = parser.parse_args()
    
    if args.test_file:
        logger.info(f"Testando pipeline em arquivo único: {args.test_file}")
        res = process_track(args.test_file, track_id="0000", subgenero="teste", rel_filepath=os.path.basename(args.test_file))
        if res:
            emb = res["embedding"]
            logger.info(f"Shape do embedding final: {emb.shape}")
        else:
            logger.error("Falha no processamento do arquivo de teste.")
    elif args.input and args.output:
        process_directory(args.input, args.output, limit_per_genre=args.limit)
    else:
        parser.print_help()
