"""
index_library.py — Indexação em lote da biblioteca musical no ChromaDB.

USO (dentro de audio_pipeline/):
    python index_library.py --root "G:/Meu Drive/Musicas"
    python index_library.py --limit 5                   # teste com 5 faixas
    python index_library.py --dry-run                   # lista o que seria feito
    python index_library.py --reindex                   # força reprocessamento
    python index_library.py --limpar-orfas              # remove faixas sem arquivo (com confirmação)

Módulos reutilizados (somente importação, sem edição):
    pipeline.process_track, vector_store, config
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── garante que o diretório do script está no path ────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

import config
import vector_store
from pipeline import process_track

# ── constantes ────────────────────────────────────────────────────────────────
EXTENSOES_SUPORTADAS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".WAV", ".MP3", ".FLAC"}
LOTE_SIZE = 10           # gravar no ChromaDB a cada N faixas
LOG_DIR = _SCRIPT_DIR / "log"
FALHAS_CSV = LOG_DIR / "indexacao_falhas.csv"
RESULTS_METADATA = _SCRIPT_DIR / "results" / "metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rel_id(filepath: Path, root: Path) -> str:
    """
    Retorna o caminho relativo no formato que o MPD aceita:
    subgenero/nome_da_faixa.mp3  (barras '/', sem barra inicial)
    Este valor é usado como ID no ChromaDB e como argumento para client.add().
    """
    rel = filepath.relative_to(root)
    return str(rel).replace("\\", "/")


def varrer_biblioteca(root: Path) -> list[Path]:
    """Percorre recursivamente e retorna todos os arquivos de áudio suportados."""
    arquivos = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {e.lower() for e in EXTENSOES_SUPORTADAS}:
            arquivos.append(p)
    return sorted(arquivos)


def ids_existentes_no_chroma(collection) -> set[str]:
    """Retorna o conjunto de IDs já presentes na coleção."""
    ids: set[str] = set()
    limit = 500
    offset = 0
    while True:
        result = collection.get(limit=limit, offset=offset, include=[])
        batch = result.get("ids", [])
        if not batch:
            break
        ids.update(batch)
        if len(batch) < limit:
            break
        offset += limit
    return ids


def verificar_mpd_conf(root: Path) -> None:
    """Avisa se o music_directory do mpd.conf não aponta para root."""
    mpd_dir_conf = config.get_music_library_root(config.MPD_CONF_PATH)
    if mpd_dir_conf is None:
        logger.warning("mpd.conf não encontrado ou sem music_directory. Não foi possível validar.")
        return
    conf_path = Path(mpd_dir_conf)
    if conf_path.resolve() != root.resolve():
        logger.warning(
            f"ATENÇÃO: music_directory no mpd.conf aponta para '{conf_path}', "
            f"mas a indexação usa '{root}'. O comando add do MPD pode falhar.\n"
            f"         Ajuste music_directory no mpd.conf para: {root}"
        )
    else:
        logger.info(f"OK: music_directory do mpd.conf bate com --root ({root}).")


def salvar_falhas(falhas: list[dict]) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    campos = ["timestamp", "filepath", "motivo"]
    modo = "a" if FALHAS_CSV.exists() else "w"
    with open(FALHAS_CSV, modo, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if modo == "w":
            writer.writeheader()
        writer.writerows(falhas)
    logger.info(f"Falhas registradas em: {FALHAS_CSV}")


def atualizar_metadata_json(novos: list[dict]) -> None:
    """Appenda os novos metadados no metadata.json (sem re-escrever os antigos)."""
    RESULTS_METADATA.parent.mkdir(exist_ok=True)
    existentes: list = []
    if RESULTS_METADATA.exists():
        try:
            with open(RESULTS_METADATA, encoding="utf-8") as f:
                existentes = json.load(f)
        except Exception:
            existentes = []

    # Indexar por track_id para evitar duplicatas
    por_id = {str(e.get("track_id")): e for e in existentes}
    for item in novos:
        por_id[str(item["track_id"])] = item

    with open(RESULTS_METADATA, "w", encoding="utf-8") as f:
        json.dump(list(por_id.values()), f, ensure_ascii=False, indent=2)


# ── limpeza de órfãs ──────────────────────────────────────────────────────────

def limpar_orfas(collection, root: Path) -> None:
    """
    Detecta IDs no ChromaDB cujo arquivo não existe mais em root.
    Só remove com confirmação explícita.
    """
    print("\n=== Verificando órfãs no ChromaDB ===")
    ids = ids_existentes_no_chroma(collection)
    orfas = []
    for track_id in ids:
        abs_path = root / track_id.replace("/", os.sep)
        if not abs_path.exists():
            orfas.append(track_id)

    if not orfas:
        print("Nenhuma órfã encontrada. Tudo ok!")
        return

    print(f"\n{len(orfas)} faixas no ChromaDB sem arquivo correspondente em '{root}':")
    for o in orfas:
        print(f"  - {o}")

    resp = input(f"\nDeseja remover estas {len(orfas)} entradas do ChromaDB? [s/N] ").strip().lower()
    if resp == "s":
        collection.delete(ids=orfas)
        print(f"{len(orfas)} entradas removidas.")
    else:
        print("Nenhuma entrada removida.")


# ── dry-run ───────────────────────────────────────────────────────────────────

def dry_run(arquivos: list[Path], ids_existentes: set[str], root: Path) -> None:
    pendentes = [f for f in arquivos if _rel_id(f, root) not in ids_existentes]
    print(f"\n=== DRY-RUN ===")
    print(f"Total de arquivos na biblioteca  : {len(arquivos)}")
    print(f"Já indexados no ChromaDB         : {len(ids_existentes)}")
    print(f"Seriam processados agora         : {len(pendentes)}")
    print("\nPrimeiros 20 que seriam indexados:")
    for f in pendentes[:20]:
        print(f"  {_rel_id(f, root)}")
    if len(pendentes) > 20:
        print(f"  ... e mais {len(pendentes)-20}")


# ── loop principal de indexação ───────────────────────────────────────────────

def indexar(
    root: Path,
    collection,
    ids_existentes: set[str],
    limit: int | None,
    reindex: bool,
) -> dict:
    arquivos = varrer_biblioteca(root)
    if limit:
        arquivos = arquivos[:limit]

    # Filtrar já indexados (a menos que --reindex)
    if not reindex:
        pendentes = [f for f in arquivos if _rel_id(f, root) not in ids_existentes]
    else:
        pendentes = arquivos
        logger.info("--reindex ativo: todas as faixas serão reprocessadas.")

    total = len(arquivos)
    n_ja_indexadas = total - len(pendentes)
    logger.info(f"Biblioteca: {total} faixas | Já indexadas: {n_ja_indexadas} | A processar: {len(pendentes)}")

    if not pendentes:
        logger.info("Nada a fazer. ChromaDB já está atualizado.")
        return {"processadas": 0, "falhas": 0, "puladas": n_ja_indexadas}

    # Lote de inserção
    lote_ids: list[str]       = []
    lote_embs: list[list]     = []
    lote_metas: list[dict]    = []
    lote_json: list[dict]     = []

    falhas: list[dict] = []
    n_ok = 0

    t_inicio = time.time()

    with tqdm(pendentes, unit="faixa", dynamic_ncols=True) as barra:
        for filepath in barra:
            track_id = _rel_id(filepath, root)
            nome_curto = filepath.name
            barra.set_description(nome_curto[:45])

            # Verificar acesso (Drive ainda não baixado)
            try:
                tamanho = filepath.stat().st_size
                if tamanho == 0:
                    raise ValueError("Arquivo com 0 bytes (provavelmente não baixado do Drive)")
            except Exception as e:
                # Tentar 1 vez após 3s
                time.sleep(3)
                try:
                    tamanho = filepath.stat().st_size
                    if tamanho == 0:
                        raise ValueError("Ainda 0 bytes na segunda tentativa")
                except Exception as e2:
                    falhas.append({
                        "timestamp": datetime.now().isoformat(),
                        "filepath": str(filepath),
                        "motivo": f"Inacessível: {e2}",
                    })
                    continue

            # Processar faixa
            subgenero = filepath.parent.name
            try:
                resultado = process_track(
                    filepath=str(filepath),
                    track_id=track_id,
                    subgenero=subgenero,
                    rel_filepath=track_id,
                )
            except Exception as e:
                falhas.append({
                    "timestamp": datetime.now().isoformat(),
                    "filepath": str(filepath),
                    "motivo": f"Erro no pipeline: {e}",
                })
                logger.warning(f"[FALHA] {nome_curto}: {e}")
                continue

            if resultado is None:
                falhas.append({
                    "timestamp": datetime.now().isoformat(),
                    "filepath": str(filepath),
                    "motivo": "pipeline retornou None (faixa corrompida ou muito curta)",
                })
                continue

            embedding: np.ndarray = resultado.pop("embedding")
            resultado["track_id"]  = track_id  # sobrescreve com o ID relativo
            resultado["filepath"]  = track_id
            resultado["subgenero"] = subgenero

            lote_ids.append(track_id)
            lote_embs.append(embedding.tolist())
            lote_metas.append({"filename": filepath.name, "subgenero": subgenero})
            lote_json.append(resultado)
            n_ok += 1

            # Gravar lote no ChromaDB
            if len(lote_ids) >= LOTE_SIZE:
                _flush(collection, lote_ids, lote_embs, lote_metas, lote_json)
                lote_ids.clear(); lote_embs.clear(); lote_metas.clear(); lote_json.clear()

            # Atualizar ETA na barra
            elapsed = time.time() - t_inicio
            velocidade = n_ok / elapsed if elapsed > 0 else 0
            barra.set_postfix(ok=n_ok, falhas=len(falhas))

    # Flush final
    if lote_ids:
        _flush(collection, lote_ids, lote_embs, lote_metas, lote_json)

    if falhas:
        salvar_falhas(falhas)

    tempo_total = time.time() - t_inicio
    return {
        "processadas": n_ok,
        "falhas": len(falhas),
        "puladas": n_ja_indexadas,
        "tempo_s": tempo_total,
    }


def _flush(collection, ids, embs, metas, jsons):
    try:
        collection.upsert(ids=ids, embeddings=embs, metadatas=metas)
        atualizar_metadata_json(jsons)
        logger.debug(f"Lote de {len(ids)} faixas gravado no ChromaDB.")
    except Exception as e:
        logger.error(f"Erro ao gravar lote no ChromaDB: {e}")


# ── relatório inicial ─────────────────────────────────────────────────────────

def relatorio_inicial(root: Path, collection) -> None:
    print("\n" + "="*60)
    print(" RELATÓRIO INICIAL — INDEXAÇÃO EM LOTE")
    print("="*60)

    arquivos = varrer_biblioteca(root)
    ids = ids_existentes_no_chroma(collection)

    print(f"  Biblioteca em           : {root}")
    print(f"  Arquivos de áudio       : {len(arquivos)}")
    print(f"  Já no ChromaDB          : {len(ids)}")
    print(f"  A indexar               : {len(arquivos) - len([f for f in arquivos if _rel_id(f, root) in ids])}")

    # Verificar IDs do ChromaDB que existem na biblioteca
    ids_na_lib = {_rel_id(f, root) for f in arquivos}
    orfas = ids - ids_na_lib
    print(f"  Órfãs no ChromaDB       : {len(orfas)}" + (" (use --limpar-orfas)" if orfas else ""))
    if orfas:
        for o in list(orfas)[:10]:
            print(f"    - {o}")
        if len(orfas) > 10:
            print(f"    ... e mais {len(orfas)-10}")

    print("="*60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Indexação em lote da biblioteca musical no ChromaDB."
    )
    parser.add_argument(
        "--root", type=str, default=r"G:\Meu Drive\Musicas",
        help="Raiz da biblioteca de músicas (padrão: G:\\Meu Drive\\Musicas)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limitar N faixas (ideal para testes)")
    parser.add_argument("--reindex", action="store_true", help="Forçar reprocessamento de faixas já indexadas")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Só lista o que seria feito, sem processar")
    parser.add_argument("--limpar-orfas", action="store_true", dest="limpar_orfas", help="Remove faixas sem arquivo (pede confirmação)")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # Verificar Drive
    if not root.exists():
        logger.error(f"A pasta '{root}' não foi encontrada. Verifique se o Google Drive está montado.")
        sys.exit(1)

    # Conectar ChromaDB
    logger.info("Conectando ao ChromaDB...")
    client = vector_store.get_client(config.CHROMA_PERSIST_DIR)
    collection = vector_store.get_or_create_collection(client, config.COLLECTION_NAME)
    logger.info(f"Coleção '{config.COLLECTION_NAME}' com {collection.count()} vetores.")

    # Verificar mpd.conf
    verificar_mpd_conf(root)

    if args.limpar_orfas:
        limpar_orfas(collection, root)
        return

    relatorio_inicial(root, collection)
    ids = ids_existentes_no_chroma(collection)

    if args.dry_run:
        arquivos = varrer_biblioteca(root)
        if args.limit:
            arquivos = arquivos[:args.limit]
        dry_run(arquivos, ids, root)
        return

    print("Iniciando indexação. Pressione Ctrl+C para pausar (progresso é salvo).\n")
    try:
        resultado = indexar(
            root=root,
            collection=collection,
            ids_existentes=ids,
            limit=args.limit,
            reindex=args.reindex,
        )
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Progresso salvo. Rode novamente para continuar.")
        return

    # Relatório final
    print("\n" + "="*60)
    print(" RELATÓRIO FINAL")
    print("="*60)
    print(f"  Processadas com sucesso : {resultado['processadas']}")
    print(f"  Falhas                  : {resultado['falhas']}")
    print(f"  Puladas (já indexadas)  : {resultado['puladas']}")
    tempo = resultado.get("tempo_s", 0)
    print(f"  Tempo total             : {tempo/60:.1f} min ({tempo:.0f}s)")
    print(f"  Total no ChromaDB agora : {collection.count()}")
    if resultado["falhas"] > 0:
        print(f"  Log de falhas           : {FALHAS_CSV}")
    print("="*60)


if __name__ == "__main__":
    main()
