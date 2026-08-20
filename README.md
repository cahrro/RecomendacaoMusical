# Recomendador Musical - Integração MPD/Cantata

Este é um sistema de recomendação musical acústica baseado em IA, que utiliza ChromaDB e HuggingFace (Modelos CLAP) para analisar e recomendar faixas similares ao que você está ouvindo no momento no servidor MPD.

## Como funciona
A aplicação roda em segundo plano observando as alterações no estado do seu player MPD. Quando uma faixa está tocando (ou ao pular de faixa), a IA processa o *fingerprint* sonoro e busca no banco de vetores as faixas mais semelhantes (sem usar metadados de texto), enfileirando automaticamente a melhor recomendação para tocar em seguida!

## Estrutura do Projeto
- `audio_pipeline/`: Código principal em Python (indexação, embeddings e integração com MPD).
- `mpd.conf.example`: Exemplo de configuração do servidor MPD no Windows.
- `iniciar_sistema.bat`: Script prático para rodar tanto o servidor MPD quanto a IA integradora.

## Instalação e Uso

1. **Configuração do MPD**
   - Instale o MPD (Music Player Daemon) para Windows.
   - Renomeie o `mpd.conf.example` para `mpd.conf` e edite o `music_directory` para apontar para sua biblioteca de áudio local (ex: `"G:/Meu Drive/Musicas"`).

2. **Ambiente Python**
   ```bash
   cd audio_pipeline
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Iniciando**
   - Utilize o script `iniciar_sistema.bat` na raiz do projeto para rodar tudo automaticamente de forma sequencial.
