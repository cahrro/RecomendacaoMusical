import logging
import subprocess
import os

logger = logging.getLogger(__name__)

def generate_fingerprint(filepath: str) -> str | None:
    """
    Gera o fingerprint acústico do arquivo chamando o fpcalc nativo.
    Retorna a string crua (inteiros separados por vírgula).
    """
    try:
        # Usa o -raw para pegar o array de inteiros (32-bit), evitando a necessidade do chromaprint.dll
        result = subprocess.run(['fpcalc', '-raw', filepath], capture_output=True, text=True, check=True)
        
        for line in result.stdout.splitlines():
            if line.startswith('FINGERPRINT='):
                return line[len('FINGERPRINT='):]
        return None
    except FileNotFoundError:
        logger.error("fpcalc não encontrado! Por favor adicione fpcalc ao PATH ou na pasta raiz do projeto.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao executar fpcalc para {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao gerar fingerprint para {filepath}: {e}")
        return None

def compare_fingerprints(fp1: str, fp2: str) -> float:
    """
    Calcula a similaridade entre dois fingerprints acústicos localmente.
    Recebe as strings em formato RAW (inteiros separados por vírgula) e calcula
    a distância de Hamming normalizada.
    Retorna um score de 0.0 a 1.0.
    """
    if not fp1 or not fp2:
        return 0.0
    try:
        # Converter string separada por vírgula em lista de inteiros
        arr1 = [int(x) for x in fp1.split(',')]
        arr2 = [int(x) for x in fp2.split(',')]
        
        min_len = min(len(arr1), len(arr2))
        if min_len == 0:
            return 0.0
            
        # Comparar a sobreposição (prefixos)
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        
        # Operação XOR bit a bit e contagem de bits 1 (diferentes)
        # abs() pois fpcalc as vezes retorna inteiros sinalizados no raw
        bits_diferentes = sum(bin((abs(a) ^ abs(b)) & 0xFFFFFFFF).count('1') for a, b in zip(arr1, arr2))
        total_bits = min_len * 32
        
        similarity = 1.0 - (bits_diferentes / total_bits)
        return float(similarity)
    except Exception as e:
        logger.error(f"Erro ao comparar fingerprints: {e}")
        return 0.0

def find_duplicate_by_fingerprint(new_fp: str, known_fingerprints: dict[str, str], threshold: float = 0.95) -> str | None:
    """
    Compara o novo fingerprint contra todos os já conhecidos ({track_id: fingerprint}).
    Retorna o track_id da primeira duplicata encontrada com score >= threshold, ou None.
    
    Nota: O threshold padrão é 0.95. Valores muito baixos podem gerar falsos positivos 
    (confundir covers, remixes ou lives com a faixa original).
    """
    if not new_fp:
        return None
        
    for track_id, known_fp in known_fingerprints.items():
        if not known_fp:
            continue
        score = compare_fingerprints(new_fp, known_fp)
        if score >= threshold:
            logger.info(f"Possível duplicata encontrada! track_id={track_id} com score {score:.3f}")
            return track_id
            
    return None
