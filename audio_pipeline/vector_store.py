import argparse
import json
import numpy as np
import chromadb
from chromadb.api.models.Collection import Collection

def get_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)

def get_or_create_collection(client: chromadb.PersistentClient, name: str = "musicas_embeddings") -> Collection:
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

def index_embeddings(collection: Collection, embeddings_path: str, metadata_path: str) -> None:
    print(f"Carregando embeddings de {embeddings_path} e metadados de {metadata_path}...")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    data = np.load(embeddings_path)
    if 'embeddings' in data:
        embeddings = data['embeddings']
    elif 'arr_0' in data:
        embeddings = data['arr_0']
    else:
        key = list(data.keys())[0]
        embeddings = data[key]

    if len(embeddings) != len(metadata_list):
        raise ValueError(f"O número de embeddings ({len(embeddings)}) não bate com o número de metadados ({len(metadata_list)}).")
    
    ids = []
    embs = []
    metadatas = []
    
    seen_ids = set()
    
    for i, meta in enumerate(metadata_list):
        track_id = str(meta.get('track_id', str(i)))
        if track_id in seen_ids:
            raise ValueError(f"Encontrado track_id duplicado: {track_id}. O track_id deve ser único.")
        seen_ids.add(track_id)
        
        ids.append(track_id)
        embs.append(embeddings[i].tolist())
        
        filtered_meta = {
            "filename": str(meta.get("filename", "")),
            "subgenero": str(meta.get("subgenero", "")),
            "n_segments": int(meta.get("n_segments", 1))
        }
        metadatas.append(filtered_meta)
    
    print(f"Indexando {len(ids)} vetores...")
    collection.upsert(
        ids=ids,
        embeddings=embs,
        metadatas=metadatas
    )
    
    print(f"Sucesso! {len(ids)} vetores indexados/atualizados.")

def query_similar(collection: Collection, query_embedding: np.ndarray, k: int = 5, exclude_id: str = None) -> list[dict]:
    n_results = k + 1 if exclude_id else k
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    parsed_results = []
    
    if not results['ids'] or not results['ids'][0]:
        return parsed_results
        
    ids = results['ids'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    
    count = 0
    for i in range(len(ids)):
        if exclude_id and ids[i] == exclude_id:
            continue
            
        parsed_results.append({
            "track_id": ids[i],
            "filename": metadatas[i].get("filename", ""),
            "distance": distances[i],
            "subgenero": metadatas[i].get("subgenero", "")
        })
        count += 1
        if count == k:
            break
            
    return parsed_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operações no vector store (ChromaDB)")
    parser.add_argument("--action", type=str, required=True, choices=["index", "query"], help="Ação a ser realizada")
    parser.add_argument("--embeddings_path", type=str, default="output/embeddings.npz", help="Caminho para embeddings.npz")
    parser.add_argument("--metadata_path", type=str, default="output/metadata.json", help="Caminho para metadata.json")
    parser.add_argument("--track_id", type=str, help="ID numérico da faixa para consultar similares (ex: '0000')")
    parser.add_argument("--k", type=int, default=5, help="Número de resultados")
    
    args = parser.parse_args()
    
    client = get_client()
    collection = get_or_create_collection(client)
    
    if args.action == "index":
        index_embeddings(collection, args.embeddings_path, args.metadata_path)
    elif args.action == "query":
        if not args.track_id:
            print("Erro: --track_id é obrigatório para --action query")
            exit(1)
            
        result = collection.get(ids=[args.track_id], include=["embeddings"])
        
        if result.get('embeddings') is None or len(result['embeddings']) == 0:
            print(f"Erro: Faixa com ID '{args.track_id}' não encontrada na coleção.")
            exit(1)
            
        query_emb = np.array(result['embeddings'][0])
        
        similares = query_similar(collection, query_emb, k=args.k, exclude_id=args.track_id)
        
        print(f"\nResultados para '{args.track_id}' (K={args.k}):")
        for i, res in enumerate(similares):
            safe_filename = res['filename'].encode('ascii', 'ignore').decode('ascii')
            print(f"{i+1}. [Dist: {res['distance']:.4f}] {safe_filename} ({res['subgenero']}) - ID: {res['track_id']}")
