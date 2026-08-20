import numpy as np
import logging
import os
import subprocess
import shlex

try:
    import acoustid
except ImportError:
    logging.warning("Módulo 'acoustid' não encontrado. Instale pyacoustid.")

try:
    import chromaprint
    HAS_CHROMAPRINT = True
except ImportError as e:
    logging.warning(f"Módulo 'chromaprint' falhou ao importar (possivelmente DLL ausente): {e}")
    HAS_CHROMAPRINT = False

logger = logging.getLogger(__name__)

def run_fpcalc_raw(filepath: str) -> str:
    """Executa o fpcalc -raw diretamente para obter o fingerprint não comprimido (fallback)."""
    try:
        # Tentar fpcalc do PATH ou do diretorio atual
        cmd = ["fpcalc", "-raw", filepath]
        if os.name == 'nt' and os.path.exists('fpcalc.exe'):
            cmd[0] = r".\fpcalc.exe"
            
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"Erro no fpcalc: {result.stderr}")
            return None
            
        for line in result.stdout.splitlines():
            if line.startswith("FINGERPRINT="):
                return line[len("FINGERPRINT="):]
        return None
    except Exception as e:
        logger.error(f"Falha ao executar fpcalc diretamente: {e}")
        return None

def generate_fingerprint(filepath: str) -> str:
    """
    Gera o fingerprint acústico do arquivo.
    Retorna a string base64 codificada se chromaprint estiver disponível, 
    ou a lista bruta de inteiros (comma-separated) se não estiver.
    """
    if HAS_CHROMAPRINT:
        try:
            duration, fp = acoustid.fingerprint_file(filepath)
            if isinstance(fp, bytes):
                return fp.decode('utf-8')
            return fp
        except Exception as e:
            logger.error(f"Erro ao gerar fingerprint para {filepath}: {e}")
            return None
    else:
        # Fallback para fpcalc -raw
        return run_fpcalc_raw(filepath)

def compare_fingerprints(fp1: str, fp2: str) -> float:
    """
    Compara dois fingerprints e retorna a similaridade (0 a 1).
    Suporta tanto base64 do chromaprint quanto raw CSV fallback.
    """
    if fp1 == fp2:
        return 1.0

    try:
        if HAS_CHROMAPRINT and not (',' in fp1 or ',' in fp2):
            if isinstance(fp1, str):
                fp1 = fp1.encode('utf-8')
            if isinstance(fp2, str):
                fp2 = fp2.encode('utf-8')
            arr1, _ = chromaprint.decode_fingerprint(fp1)
            arr2, _ = chromaprint.decode_fingerprint(fp2)
        else:
            # Assumimos que são listas CSV em fallback
            arr1 = [int(x) for x in fp1.split(',') if x.strip()]
            arr2 = [int(x) for x in fp2.split(',') if x.strip()]
    except Exception as e:
        logger.error(f"Erro ao decodificar fingerprints: {e}")
        return 0.0

    arr1 = np.array(arr1, dtype=np.uint32)
    arr2 = np.array(arr2, dtype=np.uint32)

    len1 = len(arr1)
    len2 = len(arr2)

    if len1 == 0 or len2 == 0:
        return 0.0

    if len1 > len2:
        arr1, arr2 = arr2, arr1
        len1, len2 = len2, len1

    max_sim = 0.0
        
    for offset in range(len2 - len1 + 1):
        segment2 = arr2[offset:offset+len1]
        
        xor_result = np.bitwise_xor(arr1, segment2)
        
        xor_bytes = xor_result.view(np.uint8)
        bits_diferentes = np.unpackbits(xor_bytes).sum()
        
        total_bits = len1 * 32
        sim = 1.0 - (bits_diferentes / total_bits)
        
        if sim > max_sim:
            max_sim = sim
            
        if max_sim > 0.98:
            break

    return max_sim

def find_duplicate_by_fingerprint(new_fp: str, known_fingerprints: dict, threshold: float = 0.95) -> str:
    """
    Compara o novo fingerprint contra todos os conhecidos ({track_id: fingerprint}).
    Retorna o track_id se a similaridade >= threshold.
    """
    if not new_fp:
        return None

    best_match_id = None
    best_sim = 0.0

    for track_id, known_fp in known_fingerprints.items():
        if not known_fp:
            continue
            
        sim = compare_fingerprints(new_fp, known_fp)
        if sim >= threshold and sim > best_sim:
            best_sim = sim
            best_match_id = track_id
            
    if best_match_id and best_sim >= threshold:
        logger.info(f"Duplicata encontrada: track_id={best_match_id} (similaridade={best_sim:.3f})")
        return best_match_id
        
    return None
