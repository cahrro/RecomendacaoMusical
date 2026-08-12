import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from vector_store import get_client, get_or_create_collection, query_similar

def load_ground_truth(metadata_path: str) -> Dict[str, str]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Arquivo metadata não encontrado em: {metadata_path}")
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    ground_truth = {}
    missing_subgenre = []
    
    if isinstance(metadata, dict):
        iterator = metadata.items()
    elif isinstance(metadata, list):
        iterator = []
        for i, m in enumerate(metadata):
            track_id = str(m.get('track_id', str(i)))
            iterator.append((track_id, m))
    else:
        raise ValueError("Formato de metadata não suportado. Esperado dict ou list.")
    
    for track_id, info in iterator:
        subgenero = info.get("subgenero", "").strip()
        if not subgenero:
            filename = info.get("filename", "desconhecido")
            missing_subgenre.append(f"ID: {track_id} | Arquivo: {filename}")
        else:
            ground_truth[track_id] = subgenero
            
    if missing_subgenre:
        raise AssertionError(f"Erro de sanidade: {len(missing_subgenre)} faixas estão sem subgenero preenchido. Isso não deveria acontecer com a inferência automática.")
        
    return ground_truth

def precision_at_k_for_track(collection, track_id: str, ground_truth: Dict[str, str], k: int = 5) -> float:
    result = collection.get(ids=[track_id], include=["embeddings"])
    embeddings = result.get('embeddings')
    if embeddings is None or len(embeddings) == 0:
        return 0.0
        
    query_emb = np.array(result['embeddings'][0])
    
    similares = query_similar(collection, query_emb, k=k, exclude_id=track_id)
    
    if not similares:
        return 0.0
        
    seed_subgenero = ground_truth.get(track_id)
    if not seed_subgenero:
        return 0.0
        
    matches = sum(1 for res in similares if res.get('subgenero') == seed_subgenero)
    return matches / len(similares)

def precision_at_k_average(collection, ground_truth: Dict[str, str], k: int = 5) -> float:
    scores = []
    for track_id in ground_truth.keys():
        score = precision_at_k_for_track(collection, track_id, ground_truth, k)
        scores.append(score)
    
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def random_baseline_precision_at_k(ground_truth: Dict[str, str], k: int = 5, n_trials: int = 100) -> float:
    track_ids = list(ground_truth.keys())
    if len(track_ids) <= k:
        return 0.0
        
    total_score = 0.0
    
    for seed_track_id in track_ids:
        seed_subgenero = ground_truth[seed_track_id]
        
        candidates = [t_id for t_id in track_ids if t_id != seed_track_id]
        
        track_score = 0.0
        for _ in range(n_trials):
            sampled_ids = random.sample(candidates, min(k, len(candidates)))
            matches = sum(1 for t_id in sampled_ids if ground_truth[t_id] == seed_subgenero)
            track_score += matches / len(sampled_ids)
            
        total_score += track_score / n_trials
        
    return total_score / len(track_ids)

def run_full_evaluation(collection, ground_truth: Dict[str, str], k_values: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    results = []
    
    for k in k_values:
        print(f"Avaliando para K={k}...")
        precisao_sistema = precision_at_k_average(collection, ground_truth, k)
        precisao_baseline = random_baseline_precision_at_k(ground_truth, k)
        
        ganho = 0.0
        if precisao_baseline > 0:
            ganho = ((precisao_sistema - precisao_baseline) / precisao_baseline) * 100
            
        results.append({
            "k": k,
            "precisao_sistema": precisao_sistema,
            "precisao_baseline": precisao_baseline,
            "ganho_relativo": ganho
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação do sistema de recomendação")
    parser.add_argument("--metadata", type=str, default="./output/metadata.json", help="Caminho do metadata.json")
    parser.add_argument("--k_values", type=str, default="1,3,5,10", help="Valores de K separados por vírgula")
    parser.add_argument("--output_dir", type=str, default="./results", help="Pasta de saída dos resultados")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    
    print("Carregando ground truth e checando subgêneros...")
    try:
        ground_truth = load_ground_truth(args.metadata)
    except ValueError as e:
        print(e)
        exit(1)
        
    print(f"Total de faixas válidas para avaliação: {len(ground_truth)}")
    
    client = get_client()
    collection = get_or_create_collection(client)
    
    df_results = run_full_evaluation(collection, ground_truth, k_values)
    
    csv_path = os.path.join(args.output_dir, "precisao_at_k.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")
    
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.35
    index = np.arange(len(k_values))
    
    plt.bar(index, df_results["precisao_sistema"], bar_width, label="Sistema", color="#4C72B0")
    plt.bar(index + bar_width, df_results["precisao_baseline"], bar_width, label="Baseline Aleatória", color="#DD8452")
    
    plt.xlabel("K (Número de Recomendações)", fontsize=12)
    plt.ylabel("Precisão@K Média", fontsize=12)
    plt.title("Precisão@K: Sistema vs. Baseline Aleatória", fontsize=14)
    plt.xticks(index + bar_width / 2, k_values)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, row in df_results.iterrows():
        sys_val = row["precisao_sistema"]
        base_val = row["precisao_baseline"]
        
        plt.text(i, sys_val + 0.01, f"{sys_val:.2f}", ha='center', va='bottom', fontsize=10)
        plt.text(i + bar_width, base_val + 0.01, f"{base_val:.2f}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    img_path = os.path.join(args.output_dir, "precisao_at_k.png")
    plt.savefig(img_path, dpi=300)
    print(f"Gráfico salvo em: {img_path}")
    
    print("\nResumo dos Resultados:")
    print(df_results.to_string(index=False))
    
    k_resumo = 5 if 5 in k_values else k_values[-1]
    resumo_row = df_results[df_results["k"] == k_resumo].iloc[0]
    
    print(f"\nPrecisão@{k_resumo} do sistema: {resumo_row['precisao_sistema']:.2f} vs baseline aleatória: {resumo_row['precisao_baseline']:.2f} — ganho de {resumo_row['ganho_relativo']:.0f}%")
