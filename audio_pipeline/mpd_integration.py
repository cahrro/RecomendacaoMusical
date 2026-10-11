import argparse
import json
import logging
import os
import random
import time
from mpd import MPDClient, ConnectionError, CommandError
from collections import deque
import urllib.parse
import threading
import queue

import config
import vector_store
import auto_indexer
import status_window

# --- Constantes de Configuração da Recomendação ---
HISTORY_SIZE_DEFAULT = 30
MIN_QUEUE_AHEAD = 2
CANDIDATES_K = 20
SELECTION_MODE = "random_top_n"  # "random_top_n", "first", ou "mmr"
TOP_N_VARIETY = 3
AUTO_RESUME_PLAY_ON_STOP = False  # Retomar play se o MPD parar com fila esgotada (padrão desligada)
MAX_REPLENISH_TRIES = 5  # Limite de tentativas no loop de reposição para nunca travar a thread
SESSION_HISTORY_FILE = "session_history.json"
RANDOM_SEED = None  # None para aleatório real, ou int fixo para testes determinísticos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prompt_music_library_root(default: str | None = None) -> str:
    """Solicita o caminho da biblioteca de músicas ao usuário."""
    while True:
        prompt_text = f"Digite o caminho absoluto da pasta de músicas (ou Enter para usar o padrão: {default}): "
        user_input = input(prompt_text).strip()
        
        if not user_input:
            if default and os.path.exists(default):
                return os.path.abspath(default)
            else:
                print("Caminho padrão inválido ou não fornecido. Por favor, digite um caminho válido.")
                continue
                
        if os.path.exists(user_input):
            # Opcional: perguntar se deseja salvar para a próxima vez
            save = input("Deseja salvar este caminho como padrão para a próxima execução? (s/n): ").strip().lower()
            if save == 's':
                with open("last_music_root.txt", "w", encoding="utf-8") as f:
                    f.write(os.path.abspath(user_input))
            return os.path.abspath(user_input)
        else:
            print(f"O caminho '{user_input}' não existe. Tente novamente.")

def connect_mpd(host: str, port: int) -> MPDClient:
    """Conecta ao servidor MPD com retry automático."""
    client = MPDClient()
    max_retries = 3
    base_wait = 2
    
    for attempt in range(max_retries):
        try:
            client.connect(host, port)
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = base_wait ** attempt
                logging.warning(f"Falha ao conectar ao MPD ({e}). Tentando novamente em {wait_time} segundos...")
                time.sleep(wait_time)
            else:
                logging.error("Falha ao conectar ao MPD após várias tentativas.")
                raise e

def reconnect_if_needed(client: MPDClient) -> MPDClient:
    """Reconecta automaticamente se a conexão cair."""
    try:
        client.ping()
        return client
    except (ConnectionError, BrokenPipeError, ConnectionResetError, EOFError):
        logging.warning("Conexão com MPD perdida. Tentando reconectar...")
        try:
            client.disconnect()
        except:
            pass
        return connect_mpd(config.MPD_HOST, config.MPD_PORT)

def build_file_to_id_map(metadata_path: str, music_library_root: str) -> tuple[dict, dict]:
    """Cria mapeamentos entre os caminhos relativos ao root da biblioteca MPD e os IDs do ChromaDB."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        
    file_to_id = {}
    id_to_file = {}
    
    for meta in metadata_list:
        track_id = str(meta['track_id'])
        filepath = meta['filepath']
        
        norm_filepath = os.path.normpath(filepath)
        norm_root = os.path.normpath(music_library_root)

        if os.path.isabs(norm_filepath):
            if norm_filepath.lower().startswith(norm_root.lower()):
                rel_path = os.path.relpath(norm_filepath, norm_root)
            else:
                rel_path = filepath
        else:
            abs_path = os.path.join(norm_root, norm_filepath)
            rel_path = os.path.relpath(abs_path, norm_root)
            
        rel_path = rel_path.replace('\\', '/')
        
        file_to_id[rel_path] = track_id
        id_to_file[track_id] = rel_path
        
    if file_to_id:
        example_file, example_id = list(file_to_id.items())[0]
        logging.info(f"Exemplo de mapeamento: MPD File '{example_file}' <-> Track ID '{example_id}'")
        
    return file_to_id, id_to_file

def normalize_mpd_filepath(raw_path: str, music_root: str = "") -> str:
    """Normaliza caminhos retornados pelo MPD ou URLs do Cantata para o formato relativo padrão."""
    if not raw_path:
        return ""
    # Se for URL enviada pelo Cantata usando servidor HTTP interno (ex: http://.../?file=...)
    if raw_path.startswith("http") and "?file=" in raw_path:
        parsed_url = urllib.parse.urlparse(raw_path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        if 'file' in query_params:
            raw_path = query_params['file'][0]
    
    # Decodificar URL encoding caso exista (ex: %20 -> espaço)
    decoded_path = urllib.parse.unquote(raw_path)
    norm_path = os.path.normpath(decoded_path)
    
    if music_root and os.path.isabs(norm_path):
        try:
            norm_root = os.path.normpath(music_root)
            if norm_path.lower().startswith(norm_root.lower()):
                rel_path = os.path.relpath(norm_path, norm_root)
                return rel_path.replace('\\', '/')
        except ValueError:
            pass

    return norm_path.replace('\\', '/')

def load_session_history(filepath: str = SESSION_HISTORY_FILE, max_len: int = HISTORY_SIZE_DEFAULT) -> deque:
    """Carrega o histórico de faixas persistido em disco de forma tolerante a corrupção."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    logging.info(f"Histórico de sessão carregado com {len(data)} faixas de '{filepath}'.")
                    return deque(data[-max_len:], maxlen=max_len)
        except Exception as e:
            logging.warning(f"Aviso: Não foi possível ler histórico de '{filepath}' ({e}). Iniciando novo histórico.")
    return deque(maxlen=max_len)

def save_session_history(history: deque, filepath: str = SESSION_HISTORY_FILE) -> None:
    """Salva o histórico de faixas de forma atômica para evitar corrupção em caso de encerramento."""
    try:
        temp_file = f"{filepath}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(list(history), f, ensure_ascii=False, indent=2)
        os.replace(temp_file, filepath)
    except Exception as e:
        logging.error(f"Erro ao salvar histórico de sessão em '{filepath}': {e}")

def compute_mmr_selection(candidates: list[dict], collection, play_history: deque, lambda_param: float = 0.6) -> dict | None:
    """Aplica Maximal Marginal Relevance (MMR) entre relevância ao query e diversidade com histórico."""
    if not candidates:
        return None
    import numpy as np

    cand_ids = [c['track_id'] for c in candidates]
    cand_data = collection.get(ids=cand_ids, include=["embeddings"])
    cand_embs_map = {}
    if cand_data.get('ids') and cand_data.get('embeddings') is not None:
        for cid, emb in zip(cand_data['ids'], cand_data['embeddings']):
            cand_embs_map[cid] = np.array(emb)

    hist_ids = list(play_history)[-5:]
    hist_embs = []
    if hist_ids:
        hist_data = collection.get(ids=hist_ids, include=["embeddings"])
        if hist_data.get('embeddings') is not None:
            hist_embs = [np.array(e) for e in hist_data['embeddings']]

    best_score = -float('inf')
    best_cand = candidates[0]

    for cand in candidates:
        sim_query = 1.0 - cand.get('distance', 1.0)
        emb_cand = cand_embs_map.get(cand['track_id'])
        max_sim_hist = 0.0
        if emb_cand is not None and hist_embs:
            for h_emb in hist_embs:
                dot = np.dot(emb_cand, h_emb)
                norm = (np.linalg.norm(emb_cand) * np.linalg.norm(h_emb)) + 1e-9
                cos_sim = dot / norm
                if cos_sim > max_sim_hist:
                    max_sim_hist = cos_sim

        mmr_score = (lambda_param * sim_query) - ((1.0 - lambda_param) * max_sim_hist)
        if mmr_score > best_score:
            best_score = mmr_score
            best_cand = cand

    return best_cand

def get_next_recommendation(collection, current_track_id: str, file_to_id_map: dict, id_to_file_map: dict, k: int = 20, history: list = None) -> str | None:
    """Consulta a recomendação no banco de vetores evitando histórico recente.
    Mantém assinatura original pública para compatibilidade retroativa.
    """
    res = select_next_recommendations(
        collection=collection,
        current_track_id=current_track_id,
        id_to_file_map=id_to_file_map,
        k=k,
        history=history,
        queued_files=set(),
        failed_tracks=set()
    )
    return res[0] if res else None

def select_next_recommendations(
    collection,
    current_track_id: str,
    id_to_file_map: dict,
    k: int = CANDIDATES_K,
    history: deque | list = None,
    queued_files: set = None,
    failed_tracks: set = None,
    selection_mode: str = SELECTION_MODE,
    top_n: int = TOP_N_VARIETY,
    random_seed: int | None = RANDOM_SEED,
    msg_queue: queue.Queue = None
) -> tuple[str, dict, int] | None:
    """Seleciona a melhor candidata evitando histórico, fila atual e faixas com falha."""
    if history is None:
        history = []
    if queued_files is None:
        queued_files = set()
    if failed_tracks is None:
        failed_tracks = set()

    result = collection.get(ids=[current_track_id], include=["embeddings"])
    embeddings_list = result.get('embeddings')
    if embeddings_list is None or len(embeddings_list) == 0:
        return None

    import numpy as np
    query_emb = np.array(result['embeddings'][0])

    similares = vector_store.query_similar(collection, query_emb, k=k, exclude_id=current_track_id)
    if not similares:
        return None

    hist_set = set(history)

    def filter_candidates(effective_hist_set: set) -> list[tuple[int, dict]]:
        valid = []
        for rank, rec in enumerate(similares):
            rec_id = rec['track_id']
            rec_file = id_to_file_map.get(rec_id)

            if rec_id in failed_tracks:
                continue

            if rec_id in effective_hist_set:
                logging.debug(f"Candidata '{rec_id}' rejeitada (motivo: no histórico)")
                continue

            if rec_file and (rec_file in queued_files or normalize_mpd_filepath(rec_file) in queued_files):
                logging.info(f"Candidata '{rec_file}' rejeitada (motivo: na fila)")
                if msg_queue:
                    msg_queue.put({"type": "log", "text": f"Rejeitada {os.path.basename(rec_file)} (já na fila)"})
                continue

            if rec_file:
                valid.append((rank, rec))
        return valid

    valid_candidates = filter_candidates(hist_set)

    # Esgotamento: se todas forem rejeitadas por estarem no histórico, descartar entradas mais antigas
    if not valid_candidates and hist_set:
        logging.warning("Todas as candidatas estão no histórico. Descartando entradas mais antigas do histórico...")
        if msg_queue:
            msg_queue.put({"type": "log", "text": "Histórico esgotado: relaxando histórico antigo"})
        temp_hist = list(history)
        while temp_hist and not valid_candidates:
            temp_hist.pop(0)  # remove mais antiga
            valid_candidates = filter_candidates(set(temp_hist))

    if not valid_candidates:
        return None

    # Variedade configurável
    rng = random.Random(random_seed) if random_seed is not None else random

    if selection_mode == "first" or len(valid_candidates) == 1:
        chosen_rank, chosen_rec = valid_candidates[0]
    elif selection_mode == "mmr":
        cand_list = [c for _, c in valid_candidates[:top_n]]
        chosen_rec = compute_mmr_selection(cand_list, collection, history if isinstance(history, deque) else deque(history))
        chosen_rank = next(r for r, c in valid_candidates if c['track_id'] == chosen_rec['track_id'])
    else:  # "random_top_n" (padrão)
        pool = valid_candidates[:top_n]
        chosen_rank, chosen_rec = rng.choice(pool)

    chosen_file = id_to_file_map.get(chosen_rec['track_id'])
    return chosen_file, chosen_rec, chosen_rank

def replenish_mpd_queue(
    client: MPDClient,
    collection,
    current_track_id: str,
    file_to_id_map: dict,
    id_to_file_map: dict,
    music_root: str,
    play_history: deque,
    failed_tracks: set,
    min_queue_ahead: int = MIN_QUEUE_AHEAD,
    max_tries: int = MAX_REPLENISH_TRIES,
    selection_mode: str = SELECTION_MODE,
    top_n: int = TOP_N_VARIETY,
    random_seed: int | None = RANDOM_SEED,
    msg_queue: queue.Queue = None,
    on_rec_success: callable = None,
    on_error: callable = None
) -> int:
    """Garante que a fila do MPD tenha pelo menos `min_queue_ahead` faixas à frente da atual."""
    playlist = client.playlistinfo()
    status = client.status()
    current_pos = int(status.get("song", -1))

    if current_pos == -1:
        ahead = len(playlist)
    else:
        ahead = len(playlist) - (current_pos + 1)

    added_count = 0
    attempts = 0

    while ahead < min_queue_ahead and attempts < max_tries:
        attempts += 1

        # Mapeia todas as músicas atualmente na fila (tanto normalizadas quanto originais)
        queued_files = set()
        for item in playlist:
            f = item.get("file", "")
            if f:
                queued_files.add(f)
                queued_files.add(normalize_mpd_filepath(f, music_root))

        selection = select_next_recommendations(
            collection=collection,
            current_track_id=current_track_id,
            id_to_file_map=id_to_file_map,
            k=CANDIDATES_K,
            history=play_history,
            queued_files=queued_files,
            failed_tracks=failed_tracks,
            selection_mode=selection_mode,
            top_n=top_n,
            random_seed=random_seed,
            msg_queue=msg_queue
        )

        if not selection:
            logging.warning("Não há candidatas disponíveis para repor a fila. Escutando sem interromper.")
            if msg_queue:
                msg_queue.put({"type": "log", "text": "Aviso: Sem candidatas para repor a fila"})
            break

        candidate_file, candidate_rec, candidate_rank = selection
        candidate_id = candidate_rec['track_id']
        dist = candidate_rec.get('distance', 0.0)

        try:
            client.add(candidate_file)
            logging.info(f"Recomendada '{candidate_file}' (posição #{candidate_rank + 1} na lista de candidatas, distância: {dist:.4f})")
            if msg_queue:
                msg_queue.put({"type": "recommendation", "text": f"{os.path.basename(candidate_file)} (Adicionada!)"})
                msg_queue.put({"type": "log", "text": f"Injetada: {os.path.basename(candidate_file)} (pos #{candidate_rank + 1})"})

            play_history.append(candidate_id)
            save_session_history(play_history)

            added_count += 1
            if on_rec_success:
                on_rec_success()

            # Atualiza imediatamente a fila conhecida
            queued_files.add(candidate_file)
            queued_files.add(normalize_mpd_filepath(candidate_file, music_root))

            playlist = client.playlistinfo()
            ahead = len(playlist) - (current_pos + 1) if current_pos != -1 else len(playlist)

        except CommandError as e:
            logging.error(f"Candidata '{candidate_file}' rejeitada (motivo: add falhou - {e})")
            failed_tracks.add(candidate_id)
            if on_error:
                on_error(e)
            if msg_queue:
                msg_queue.put({"type": "recommendation", "text": f"{os.path.basename(candidate_file)} (Falhou)"})
                msg_queue.put({"type": "log", "text": f"Candidata rejeitada (add falhou): {os.path.basename(candidate_file)}"})

    return added_count

def monitor_and_recommend(client: MPDClient, collection, file_to_id_map: dict, id_to_file_map: dict, trigger_mode: str = "on_change", music_root: str = "", metadata_path: str = "", known_fingerprints: dict = None, msg_queue: queue.Queue = None) -> None:
    """Loop principal que aguarda mudanças no MPD e enfileira recomendações com garantia de reposição."""
    if known_fingerprints is None:
        known_fingerprints = {}
    last_processed_file = None

    # Limitar HISTORY_SIZE a no máximo 50% das faixas indexadas
    total_indexed = max(1, len(file_to_id_map))
    effective_history_size = min(HISTORY_SIZE_DEFAULT, max(2, total_indexed // 2))
    logging.info(f"Tamanho do histórico de sessão configurado: {effective_history_size} faixas (total indexado: {total_indexed}).")

    play_history = load_session_history(SESSION_HISTORY_FILE, max_len=effective_history_size)
    failed_tracks = set()

    cnt_recs = 0
    cnt_idxs = 0
    cnt_errs = 0

    def update_counters():
        if msg_queue:
            msg_queue.put({"type": "counters", "recs": cnt_recs, "idxs": cnt_idxs, "errs": cnt_errs})

    def handle_rec_success():
        nonlocal cnt_recs
        cnt_recs += 1
        update_counters()

    def handle_err(e):
        nonlocal cnt_errs
        cnt_errs += 1
        update_counters()

    if msg_queue:
        msg_queue.put({"type": "status", "state": "monitorando"})
        update_counters()
    
    while True:
        try:
            client = reconnect_if_needed(client)
            
            # O client.idle("player") bloqueia até que ocorra um evento de player
            events = client.idle("player")
            
            if "player" in events:
                status = client.status()
                state = status.get("state")
                
                if state == "play":
                    current_song = client.currentsong()
                    raw_file = current_song.get("file")
                    if not raw_file:
                        continue
                        
                    current_file = normalize_mpd_filepath(raw_file, music_root)
                    
                    if current_file == last_processed_file:
                        continue
                        
                    logging.info(f"Usuário ouvindo: {current_file}")
                    if msg_queue:
                        msg_queue.put({"type": "track", "name": os.path.basename(current_file)})
                        msg_queue.put({"type": "event", "name": "Faixa Trocada / Play"})
                        msg_queue.put({"type": "log", "text": f"Ouvindo: {os.path.basename(current_file)}"})
                    
                    last_processed_file = current_file
                    current_track_id = file_to_id_map.get(current_file)
                    
                    if not current_track_id:
                        logging.warning(f"Faixa atual '{current_file}' não encontrada no índice. Tentando indexação automática...")
                        if msg_queue:
                            msg_queue.put({"type": "status", "state": "indexando"})
                            msg_queue.put({"type": "log", "text": "Indexando nova faixa..."})
                        try:
                            abs_filepath = current_file if os.path.isabs(current_file) else os.path.join(music_root, current_file)
                            if not os.path.exists(abs_filepath):
                                logging.error(f"Arquivo não encontrado fisicamente: {abs_filepath}")
                                continue
                                
                            new_track_id = auto_indexer.index_unknown_track(
                                filepath=abs_filepath,
                                metadata_path=metadata_path,
                                collection=collection,
                                known_fingerprints=known_fingerprints,
                                music_library_root=music_root
                            )
                            
                            if new_track_id:
                                file_to_id_map[current_file] = new_track_id
                                id_to_file_map[new_track_id] = current_file
                                current_track_id = new_track_id
                                cnt_idxs += 1
                                update_counters()
                                if msg_queue:
                                    msg_queue.put({"type": "log", "text": "Faixa indexada com sucesso."})
                            else:
                                logging.warning("Falha na indexação automática. Nenhuma recomendação será gerada.")
                                if msg_queue:
                                    msg_queue.put({"type": "log", "text": "Falha na indexação."})
                                continue
                        finally:
                            if msg_queue:
                                msg_queue.put({"type": "status", "state": "monitorando"})
                    
                    # Registrar no histórico e salvar
                    if current_track_id not in play_history:
                        play_history.append(current_track_id)
                        save_session_history(play_history)
                    
                    # Garantir que a fila do MPD tenha pelo menos MIN_QUEUE_AHEAD músicas à frente
                    replenish_mpd_queue(
                        client=client,
                        collection=collection,
                        current_track_id=current_track_id,
                        file_to_id_map=file_to_id_map,
                        id_to_file_map=id_to_file_map,
                        music_root=music_root,
                        play_history=play_history,
                        failed_tracks=failed_tracks,
                        min_queue_ahead=MIN_QUEUE_AHEAD,
                        max_tries=MAX_REPLENISH_TRIES,
                        msg_queue=msg_queue,
                        on_rec_success=handle_rec_success,
                        on_error=handle_err
                    )

                elif state == "stop":
                    # Fila esgotada ou usuário parou a reprodução
                    playlist = client.playlistinfo()
                    if len(playlist) == 0:
                        logging.warning("MPD em estado 'stop' com fila vazia. Repondo faixas na fila...")
                        if msg_queue:
                            msg_queue.put({"type": "log", "text": "MPD parado (fila vazia). Repondo fila..."})

                        # Usa a última faixa do histórico ou alguma faixa conhecida para repor
                        last_id = play_history[-1] if play_history else list(id_to_file_map.keys())[0] if id_to_file_map else None
                        if last_id:
                            replenish_mpd_queue(
                                client=client,
                                collection=collection,
                                current_track_id=last_id,
                                file_to_id_map=file_to_id_map,
                                id_to_file_map=id_to_file_map,
                                music_root=music_root,
                                play_history=play_history,
                                failed_tracks=failed_tracks,
                                min_queue_ahead=MIN_QUEUE_AHEAD,
                                max_tries=MAX_REPLENISH_TRIES,
                                msg_queue=msg_queue,
                                on_rec_success=handle_rec_success,
                                on_error=handle_err
                            )
                            # Se a constante permitir, retoma a reprodução automaticamente
                            if AUTO_RESUME_PLAY_ON_STOP:
                                try:
                                    client.play(0)
                                    logging.info("Reprodução retomada automaticamente após reposição da fila.")
                                except Exception as e:
                                    logging.error(f"Erro ao retomar reprodução: {e}")
                        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.error(f"Erro no ciclo de monitoramento: {e}")
            cnt_errs += 1
            update_counters()
            if msg_queue:
                msg_queue.put({"type": "status", "state": "erro"})
                msg_queue.put({"type": "log", "text": f"Erro: {e}"})
                msg_queue.put({"type": "status", "state": "monitorando"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integração do MPD com o Recomendador")
    parser.add_argument("--metadata", type=str, required=True, help="Caminho do metadata.json")
    parser.add_argument("--k", type=int, default=1, help="Quantidade de recomendações a consultar")
    parser.add_argument("--trigger_mode", type=str, choices=["on_change", "near_end"], default="on_change")
    parser.add_argument("--music_root", type=str, help="Caminho absoluto da biblioteca (pula prompt)")
    
    args = parser.parse_args()
    
    # 1. Determinar raiz da biblioteca
    if args.music_root:
        if not os.path.exists(args.music_root):
            logging.error(f"O caminho --music_root fornecido não existe: {args.music_root}")
            exit(1)
        music_root = os.path.abspath(args.music_root)
    else:
        # Tenta carregar do last_music_root.txt primeiro
        default_root = None
        if os.path.exists("last_music_root.txt"):
            with open("last_music_root.txt", "r", encoding="utf-8") as f:
                default_root = f.read().strip()
        
        # Se não houver, tenta do mpd.conf
        if not default_root:
            default_root = config.get_music_library_root(config.MPD_CONF_PATH)
            
        try:
            music_root = prompt_music_library_root(default_root)
        except KeyboardInterrupt:
            logging.info("Interrompido pelo usuário.")
            exit(0)
            
    # 2. Carregar metadados e mapear caminhos
    logging.info("Construindo mapeamento de arquivos...")
    file_to_id, id_to_file = build_file_to_id_map(args.metadata, music_root)
    
    if not file_to_id:
        logging.error("Nenhum arquivo mapeado a partir do metadata.json!")
        exit(1)
        
    logging.info(f"Total de {len(file_to_id)} faixas mapeadas.")
    
    # Valida se pelo menos 1 arquivo mapeado realmente existe no sistema de arquivos para avisar o usuário
    # Isso é apenas informativo
    valid_files_count = 0
    for rel_path in file_to_id.keys():
        abs_p = os.path.join(music_root, rel_path)
        if os.path.exists(abs_p):
            valid_files_count += 1
            break
            
    if valid_files_count == 0:
        logging.warning("CUIDADO: O mapeamento foi criado, mas o primeiro arquivo testado não foi encontrado fisicamente na pasta base.")
        logging.warning("Pode haver um problema na definição do diretório de músicas ou na normalização.")
        # Não damos exit pois o MPD pode estar em outra máquina onde o caminho é válido, mas localmente não,
        # ou se usou um diretório virtual.
        
    # Carregar fingerprints
    logging.info("Carregando fingerprints conhecidos...")
    known_fingerprints = auto_indexer.load_known_fingerprints(args.metadata)
    
    # 3. Conectar ChromaDB
    chroma_client = vector_store.get_client(config.CHROMA_PERSIST_DIR)
    collection = vector_store.get_or_create_collection(chroma_client, config.COLLECTION_NAME)
    
    # 4. Conectar MPD
    try:
        logging.info(f"Conectando ao MPD em {config.MPD_HOST}:{config.MPD_PORT}...")
        mpd_client = connect_mpd(config.MPD_HOST, config.MPD_PORT)
        logging.info("Conectado com sucesso.")
        
        print("\n=============================================")
        print("Sistema de Recomendação Ativo e Monitorando...")
        print("=============================================\n")
        
        msg_queue = queue.Queue()
        
        # 5. Loop principal em Background Thread
        mpd_thread = threading.Thread(target=monitor_and_recommend, args=(mpd_client, collection, file_to_id, id_to_file, args.trigger_mode, music_root, args.metadata, known_fingerprints, msg_queue), daemon=True)
        mpd_thread.start()
        
        # 6. Iniciar Janela de Status na Thread Principal
        win = status_window.StatusWindow(msg_queue)
        win.start()
        
    except KeyboardInterrupt:
        logging.info("\nEncerrando sistema (KeyboardInterrupt)...")
    finally:
        try:
            if 'mpd_client' in locals():
                mpd_client.disconnect()
        except:
            pass
