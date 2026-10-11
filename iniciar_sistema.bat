@echo off
:: ============================================================
:: iniciar_sistema.bat
:: Sobe o sistema completo de recomendação musical:
::   1. Aguarda o Google Drive montar a unidade G:\
::   2. Inicia o servidor MPD em janela separada
::   3. (Opcional) Inicia o painel Streamlit de métricas
::   4. Inicia o integrador Python (recomendador ativo)
::
:: Para parar o painel: execute audio_pipeline\results\parar_painel.bat
:: ============================================================

:: --- CONFIGURAÇÃO: defina 1 para abrir o painel, 0 para não abrir ---
set ABRIR_PAINEL=1

:: --- CONFIGURAÇÃO: defina 1 para indexar faixas novas ao iniciar (mais lento), 0 para desativar ---
set INDEXAR_NOVAS=0

echo ===================================================
echo Iniciando Sistema de Recomendacao Musical...
echo ===================================================

echo.
echo [1/3] Verificando acesso a pasta do Google Drive...
:WAIT_DRIVE
if not exist "G:\Meu Drive\Musicas\" (
    echo Aguardando o Google Drive montar a unidade G:...
    timeout /t 5 /nobreak >nul
    goto WAIT_DRIVE
)
echo Google Drive montado com sucesso.

echo Iniciando o Servidor de Musica (MPD) em uma nova janela...
start "Servidor MPD" cmd /c "C:\mpd\mpd.exe C:\mpd\mpd.conf"

echo [2/4] Indexando faixas novas na biblioteca...
if "%INDEXAR_NOVAS%"=="1" (
    start "Indexador" /min cmd /k "cd /d "%~dp0audio_pipeline" && .\venv\Scripts\python.exe index_library.py --root "G:\Meu Drive\Musicas""
    echo Indexacao iniciada em segundo plano ^(janela minimizada^).
) else (
    echo Indexacao de novas faixas desativada ^(INDEXAR_NOVAS=0^).
)

echo.
echo [3/4] Iniciando o Painel de Metricas ^(Streamlit^)...
if "%ABRIR_PAINEL%"=="1" (
    start "" /b cmd /c ""%~dp0audio_pipeline\results\abrir_painel.bat" >nul 2>&1"
    echo Painel iniciando em segundo plano. Acesse: http://localhost:8501
    echo Se nao abrir sozinho, execute: audio_pipeline\results\abrir_painel.bat
) else (
    echo Painel desativado ^(ABRIR_PAINEL=0^). Para ativar, edite este arquivo.
)

echo.
echo [4/4] Iniciando o Integrador (Python)...
cd "%~dp0audio_pipeline"
.\venv\Scripts\python.exe mpd_integration.py --metadata results\metadata.json --music_root "G:\Meu Drive\Musicas"

echo.
echo O programa encerrou.
echo Para parar o painel Streamlit, execute: audio_pipeline\results\parar_painel.bat
pause
