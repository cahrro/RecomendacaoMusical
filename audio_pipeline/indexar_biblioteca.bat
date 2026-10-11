@echo off
:: ============================================================
:: indexar_biblioteca.bat
:: Executa a indexação em lote da biblioteca musical no ChromaDB.
::
:: USO: Duplo clique para indexar faixas novas.
::      Edite LIMIT abaixo para testar com poucas faixas primeiro.
::
:: Para opções avançadas, rode manualmente:
::   .\venv\Scripts\python.exe index_library.py --help
:: ============================================================

:: --- Configuração: 0 = indexar tudo, N = testar com N faixas ---
set LIMIT=0

:: --- Caminhos relativos ao .bat (funciona mesmo se mover o projeto) ---
set "PIPELINE_DIR=%~dp0"
:: Remove barra final
if "%PIPELINE_DIR:~-1%" == "\" set "PIPELINE_DIR=%PIPELINE_DIR:~0,-1%"
set "VENV_PYTHON=%PIPELINE_DIR%\venv\Scripts\python.exe"
set "SCRIPT=%PIPELINE_DIR%\index_library.py"
set "MUSIC_ROOT=G:\Meu Drive\Musicas"

echo ===================================================
echo  Indexador de Biblioteca Musical - ChromaDB
echo ===================================================
echo.

:: --- Verificar venv ---
if not exist "%VENV_PYTHON%" (
    echo [ERRO] Ambiente virtual Python nao encontrado em:
    echo   %VENV_PYTHON%
    echo.
    echo Execute primeiro:
    echo   cd audio_pipeline
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: --- Verificar script ---
if not exist "%SCRIPT%" (
    echo [ERRO] Script index_library.py nao encontrado em:
    echo   %SCRIPT%
    pause
    exit /b 1
)

:: --- Verificar Google Drive ---
if not exist "%MUSIC_ROOT%\" (
    echo [ERRO] Pasta da biblioteca nao encontrada: %MUSIC_ROOT%
    echo Verifique se o Google Drive esta montado na unidade G:.
    pause
    exit /b 1
)

echo Biblioteca : %MUSIC_ROOT%
echo Python     : %VENV_PYTHON%
echo.

if "%LIMIT%" == "0" (
    echo Modo: COMPLETO ^(todas as faixas novas^)
    "%VENV_PYTHON%" "%SCRIPT%" --root "%MUSIC_ROOT%"
) else (
    echo Modo: TESTE com %LIMIT% faixas
    "%VENV_PYTHON%" "%SCRIPT%" --root "%MUSIC_ROOT%" --limit %LIMIT%
)

echo.
echo Concluido.
pause
