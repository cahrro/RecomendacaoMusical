import argparse
import json
import logging
import os
import time
from mpd import MPDClient, ConnectionError, CommandError
from collections import deque
import urllib.parse

import config
import vector_store

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
        
        if os.path.isabs(filepath):
            if filepath.startswith(music_library_root):
                rel_path = os.path.relpath(filepath, music_library_root)
            else:
                rel_path = filepath
        else:
            abs_path = os.path.join(music_library_root, filepath)
            rel_path = os.path.relpath(abs_path, music_library_root)
            
        rel_path = rel_path.replace('\\', '/')
        
        file_to_id[rel_path] = track_id
        id_to_file[track_id] = rel_path
        
    if file_to_id:
        example_file, example_id = list(file_to_id.items())[0]
        logging.info(f"Exemplo de mapeamento: MPD File '{example_file}' <-> Track ID '{example_id}'")
        
    return file_to_id, id_to_file

def get_next_recommendation(collection, current_track_id: str, file_to_id_map: dict, id_to_file_map: dict, k: int = 20, history: list = None) -> str | None:
    """Consulta a recomendação no banco de vetores evitando histórico recente."""
    result = collection.get(ids=[current_track_id], include=["embeddings"])
    embeddings_list = result.get('embeddings')
    if embeddings_list is None or len(embeddings_list) == 0:
        return None
        
    import numpy as np
    query_emb = np.array(result['embeddings'][0])
    
    similares = vector_store.query_similar(collection, query_emb, k=k, exclude_id=current_track_id)
    
    if similares:
        for rec in similares:
            rec_id = rec['track_id']
            if history and rec_id in history:
                continue
            if rec_id in id_to_file_map:
                return id_to_file_map[rec_id]
            
    return None

def monitor_and_recommend(client: MPDClient, collection, file_to_id_map: dict, id_to_file_map: dict, trigger_mode: str = "on_change", music_root: str = "") -> None:
    """Loop principal que aguarda mudanças no MPD e enfileira recomendações."""
    last_processed_file = None
    play_history = deque(maxlen=15)
    
    while True:
        try:
            client = reconnect_if_needed(client)
            
            # O client.idle("player") bloqueia até que ocorra um evento de player
            events = client.idle("player")
            
            if "player" in events:
                status = client.status()
                
                if status.get("state") == "play":
                    current_song = client.currentsong()
                    current_file = current_song.get("file")
                    
                    if not current_file:
                        continue
                        
                    cantata_base_url = None
                    # Tratamento para URLs enviadas pelo cliente Cantata usando seu servidor HTTP interno
                    if current_file.startswith("http") and "?file=" in current_file:
                        parsed_url = urllib.parse.urlparse(current_file)
                        cantata_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
                        query_params = urllib.parse.parse_qs(parsed_url.query)
                        if 'file' in query_params:
                            abs_path = query_params['file'][0]
                            abs_path = os.path.normpath(abs_path)
                            try:
                                rel_path = os.path.relpath(abs_path, music_root)
                                current_file = rel_path.replace('\\', '/')
                            except ValueError:
                                current_file = abs_path.replace('\\', '/')
                    
                    if current_file == last_processed_file:
                        continue
                        
                    logging.info(f"Usuário ouvindo: {current_file}")
                    
                    # Checar trigger mode
                    if trigger_mode == "near_end":
                        elapsed = float(status.get("elapsed", 0))
                        duration = float(status.get("duration", 0))
                        if duration - elapsed > 15:
                            pass
                    
                    last_processed_file = current_file
                    
                    current_track_id = file_to_id_map.get(current_file)
                    
                    if not current_track_id:
                        logging.warning(f"Faixa atual '{current_file}' não encontrada no índice.")
                        continue
                        
                    play_history.append(current_track_id)
                        
                    next_file = get_next_recommendation(collection, current_track_id, file_to_id_map, id_to_file_map, k=20, history=play_history)
                    
                    if next_file:
                        # Verificar se já não está na fila
                        playlist = client.playlistinfo()
                        current_pos = int(status.get("song", -1))
                        
                        already_in_queue = False
                        if current_pos != -1 and current_pos + 1 < len(playlist):
                            next_song = playlist[current_pos + 1]
                            next_song_file = next_song.get("file", "")
                            # Pode estar na fila como path relativo ou como URL do cantata
                            if next_song_file == next_file or next_file in urllib.parse.unquote(next_song_file):
                                already_in_queue = True
                                
                        if already_in_queue:
                            logging.info(f"Recomendação '{next_file}' já é a próxima na fila. Pulando.")
                        else:
                            try:
                                if cantata_base_url:
                                    abs_next_path = os.path.join(music_root, next_file).replace('\\', '/')
                                    abs_next_path_encoded = urllib.parse.quote(abs_next_path)
                                    filename_encoded = urllib.parse.quote(os.path.basename(next_file))
                                    path_to_add = f"{cantata_base_url}{filename_encoded}?file={abs_next_path_encoded}&id=-1&cantata=song"
                                else:
                                    path_to_add = next_file
                                    
                                # Adiciona a recomendação no final da fila (ou após a música atual se preferir gerenciar ordem exata)
                                client.add(path_to_add)
                                logging.info(f"IA recomendou e injetou na fila: {next_file}")
                            except CommandError as e:
                                logging.error(f"Erro ao adicionar '{next_file}' na fila: {e}")
                    else:
                        logging.warning("Não foi possível gerar uma recomendação.")
                        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.error(f"Erro no ciclo de monitoramento: {e}")
            time.sleep(2) # Pausa curta antes de tentar novamente

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
        
        # 5. Loop principal
        monitor_and_recommend(mpd_client, collection, file_to_id, id_to_file, args.trigger_mode, music_root)
        
    except KeyboardInterrupt:
        logging.info("\nEncerrando sistema (KeyboardInterrupt)...")
    finally:
        try:
            if 'mpd_client' in locals():
                mpd_client.disconnect()
        except:
            pass
