# Sistema de Recomendação de Músicas

Este projeto utiliza ferramentas de Machine Learning e processamento de áudio para recomendar músicas com base no som (embeddings gerados por CLAP) e metadados.

## Dependências Adicionais (Windows)

O sistema utiliza a biblioteca `pyacoustid` para gerar "fingerprints" acústicos das músicas, permitindo identificar duplicatas e otimizar o banco de dados.

Para que o `pyacoustid` funcione corretamente no Windows, é **necessário** instalar o binário `fpcalc` (parte do projeto Chromaprint):

1. Baixe o Chromaprint para Windows no site oficial: [https://acoustid.org/chromaprint](https://acoustid.org/chromaprint)
2. Extraia o arquivo ZIP baixado.
3. Copie o arquivo `fpcalc.exe` para uma pasta que esteja no seu `PATH` de variáveis de ambiente do Windows, ou adicione a pasta extraída ao `PATH`.
4. (Alternativa) Você pode colocar o `fpcalc.exe` diretamente na raiz do projeto (mesma pasta deste README) ou na pasta `audio_pipeline/`.

Sem o `fpcalc.exe`, a identificação de músicas duplicadas ou desconhecidas irá falhar.
