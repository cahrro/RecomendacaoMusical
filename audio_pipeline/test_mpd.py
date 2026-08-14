# from mpd import MPDClient
# import time

# def conectar_mpd():
#     client = MPDClient()
#     client.connect("localhost", 6600)
#     return client

# def logic_de_recomendacao(caminho_musica_atual):
#     # Aqui entra o código dos alunos:
#     # 1. Recuperar o embedding de 'caminho_musica_atual'
#     # 2. Buscar no Banco Vetorial a música mais similar (similaridade de cosseno)
#     # 3. Retornar o caminho do arquivo da recomendação
#     return "caminho/da/proxima_musica_similar.mp3"

# def main():
#     client = conectar_mpd()
#     print("Sistema de Recomendação Ativo e Monitorando...")

#     try:
#         while True:
#             # O script para aqui e espera o MPD enviar um sinal de mudança no player
#             events = client.idle("player")
            
#             print(events)
            
#             if "player" in events:
#                 status = client.status()
#                 print(status)
                
#                 # Verifica se há algo tocando
#                 if status.get("state") == "play":
#                     current_song = client.currentsong()
#                     file_path = current_song.get("file")
                    
#                     print(current_song)
                    
#                     print(f"Usuário ouvindo: {file_path}")
                    
#                     # Gera a recomendação via IA
#                     # proxima = logic_de_recomendacao(file_path)
                    
#                     # Adiciona à fila sem interromper a música atual
#                     # client.add(proxima)
#                     # print(f"IA injetou na fila: {proxima}")

                    
#     except Exception as e:
#         print(f"Erro na conexão: {e}")
#     finally:
#         client.disconnect()

# if __name__ == "__main__":
#     main()

from mpd import MPDClient
import time

def conectar_mpd():
    client = MPDClient()
    client.connect("localhost", 6600)
    return client

def monitorar_sequencia():
    client = conectar_mpd()
    
    try:
        # 1. Listar as músicas na playlist atual
        playlist = client.playlistinfo()
        if not playlist:
            print("A playlist está vazia.")
            return

        print("--- Músicas na Playlist ---")
        for i, musica in enumerate(playlist):
            # Tenta pegar o título, se não houver, pega o nome do arquivo
            nome = musica.get("title", musica.get("file"))
            print(f"{i}: {nome}")
        
        # Identifica o primeiro e o último índice
        primeiro_index = 0
        ultimo_index = len(playlist) - 1
        
        if ultimo_index < 0:
            return

        # 2. Começar a tocar a primeira música
        print(f"\nIniciando a primeira música: {playlist[primeiro_index].get('file')}")
        client.play(primeiro_index)
        
        # 3. Monitorar até que a primeira música termine
        # O estado 'idle' com 'player' bloqueia o script até haver mudança no player
        while True:
            status = client.status()
            state = status.get("state")
            current_pos = status.get("song") # Índice da música atual na playlist

            # Se o player parou (música acabou) ou mudou de música (pulou para a próxima)
            if state == "stop" or (current_pos is not None and int(current_pos) != primeiro_index):
                print("\nA primeira música terminou ou foi alterada.")
                break
            
            # Pequeno delay para não sobrecarregar o processador se o idle não for suportado
            client.idle("player") 

        # 4. Tocar a última música da lista
        print(f"Tocando a última música da playlist: {playlist[ultimo_index].get('file')}")
        client.play(ultimo_index)

    except Exception as e:
        print(f"Erro na execução: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    monitorar_sequencia()