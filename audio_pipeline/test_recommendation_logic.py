import os
import sys
import unittest
from collections import deque
from unittest.mock import MagicMock
import numpy as np
from mpd import CommandError

# Adiciona o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mpd_integration

class FakeCollection:
    def __init__(self, tracks_data):
        """
        tracks_data: dict no formato {track_id: {"embedding": np.array, "neighbors": [rec1, rec2, ...]}}
        """
        self.tracks_data = tracks_data

    def get(self, ids, include=None):
        out_ids = []
        out_embs = []
        for tid in ids:
            if tid in self.tracks_data:
                out_ids.append(tid)
                out_embs.append(self.tracks_data[tid]["embedding"].tolist())
        return {"ids": out_ids, "embeddings": out_embs if out_embs else None}

class TestRecommendationLogic(unittest.TestCase):

    def setUp(self):
        # Mapeamentos de teste
        self.id_to_file = {
            "A": "rock/track_a.mp3",
            "B": "rock/track_b.mp3",
            "C": "rock/track_c.mp3",
            "D": "rock/track_d.mp3",
            "E": "rock/track_e.mp3",
        }
        self.file_to_id = {v: k for k, v in self.id_to_file.items()}

        # Cria embeddings sintéticos
        self.tracks_data = {
            "A": {"embedding": np.array([1.0, 0.0, 0.0])},
            "B": {"embedding": np.array([0.9, 0.1, 0.0])},
            "C": {"embedding": np.array([0.8, 0.2, 0.0])},
            "D": {"embedding": np.array([0.7, 0.3, 0.0])},
            "E": {"embedding": np.array([0.6, 0.4, 0.0])},
        }
        self.collection = FakeCollection(self.tracks_data)

    def test_mutual_neighbors_no_repetition_in_window(self):
        """Simula vizinhos mútuos (A <-> B) e verifica que a sequência não repete dentro da janela do histórico."""
        history = deque(maxlen=10)
        history.append("A")

        # Mock vector_store.query_similar para simular que para A o mais próximo é B, e para B o mais próximo é A
        original_query = mpd_integration.vector_store.query_similar
        try:
            def mock_query(col, emb, k=5, exclude_id=None):
                if exclude_id == "A":
                    return [
                        {"track_id": "B", "distance": 0.05, "filename": "track_b.mp3"},
                        {"track_id": "C", "distance": 0.10, "filename": "track_c.mp3"},
                        {"track_id": "D", "distance": 0.20, "filename": "track_d.mp3"}
                    ]
                elif exclude_id == "B":
                    return [
                        {"track_id": "A", "distance": 0.05, "filename": "track_a.mp3"},
                        {"track_id": "C", "distance": 0.10, "filename": "track_c.mp3"},
                        {"track_id": "D", "distance": 0.20, "filename": "track_d.mp3"}
                    ]
                return []

            mpd_integration.vector_store.query_similar = mock_query

            # 1. Recomendação a partir de A: deve escolher B (pois A está no histórico)
            res_a = mpd_integration.select_next_recommendations(
                collection=self.collection,
                current_track_id="A",
                id_to_file_map=self.id_to_file,
                history=history,
                selection_mode="first"
            )
            self.assertIsNotNone(res_a)
            self.assertEqual(res_a[0], "rock/track_b.mp3")

            # Registra B no histórico (como tocada/injetada)
            history.append("B")

            # 2. Recomendação a partir de B: A e B estão no histórico, então NÃO pode ser A! Deve ser C.
            res_b = mpd_integration.select_next_recommendations(
                collection=self.collection,
                current_track_id="B",
                id_to_file_map=self.id_to_file,
                history=history,
                selection_mode="first"
            )
            self.assertIsNotNone(res_b)
            self.assertEqual(res_b[0], "rock/track_c.mp3")
            self.assertNotIn(res_b[1]["track_id"], ["A", "B"])

        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_candidate_in_queue_cantata_url_and_relative_path(self):
        """Verifica se candidata já na fila (como path relativo ou URL do Cantata) é devidamente rejeitada."""
        original_query = mpd_integration.vector_store.query_similar
        try:
            mpd_integration.vector_store.query_similar = lambda col, emb, k=5, exclude_id=None: [
                {"track_id": "B", "distance": 0.05, "filename": "track_b.mp3"},
                {"track_id": "C", "distance": 0.10, "filename": "track_c.mp3"},
            ]

            # Simula que 'B' já está na fila como URL do Cantata
            cantata_url_b = "http://127.0.0.1:4567/?file=G%3A%5CMeu%20Drive%5CMusicas%5Crock%5Ctrack_b.mp3"
            queued_files = {cantata_url_b, mpd_integration.normalize_mpd_filepath(cantata_url_b, "G:/Meu Drive/Musicas")}

            res = mpd_integration.select_next_recommendations(
                collection=self.collection,
                current_track_id="A",
                id_to_file_map=self.id_to_file,
                history=[],
                queued_files=queued_files,
                selection_mode="first"
            )
            # Como B está na fila pelo formato URL do Cantata, deve pular B e escolher C
            self.assertIsNotNone(res)
            self.assertEqual(res[0], "rock/track_c.mp3")

        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_replenish_queue_handles_add_failure_and_continues(self):
        """Simula falha no client.add (ex: CommandError) e verifica que a candidata inválida é pulada e a próxima é adicionada."""
        original_query = mpd_integration.vector_store.query_similar
        try:
            mpd_integration.vector_store.query_similar = lambda col, emb, k=5, exclude_id=None: [
                {"track_id": "B", "distance": 0.05, "filename": "track_b.mp3"},
                {"track_id": "C", "distance": 0.10, "filename": "track_c.mp3"},
                {"track_id": "D", "distance": 0.20, "filename": "track_d.mp3"},
            ]

            mock_client = MagicMock()
            # Fila tem a música atual (índice 0), 0 músicas à frente
            mock_client.status.return_value = {"song": "0", "state": "play"}
            mock_client.playlistinfo.side_effect = [
                [{"file": "rock/track_a.mp3"}], # estado inicial: 0 à frente
                [{"file": "rock/track_a.mp3"}, {"file": "rock/track_c.mp3"}], # após adicionar C: 1 à frente
                [{"file": "rock/track_a.mp3"}, {"file": "rock/track_c.mp3"}, {"file": "rock/track_d.mp3"}] # após adicionar D: 2 à frente
            ]

            # B falha no add, mas C e D têm sucesso
            def mock_add(filepath):
                if "track_b.mp3" in filepath:
                    raise CommandError("Access denied or file not found")
                return None

            mock_client.add.side_effect = mock_add

            history = deque(maxlen=10)
            failed = set()

            added = mpd_integration.replenish_mpd_queue(
                client=mock_client,
                collection=self.collection,
                current_track_id="A",
                file_to_id_map=self.file_to_id,
                id_to_file_map=self.id_to_file,
                music_root="",
                play_history=history,
                failed_tracks=failed,
                min_queue_ahead=2,
                max_tries=5,
                selection_mode="first"
            )

            # B falhou e foi parar em failed_tracks
            self.assertIn("B", failed)
            # C e D foram adicionados com sucesso
            self.assertEqual(added, 2)
            self.assertIn("C", history)
            self.assertIn("D", history)

        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_exhaustion_small_library_relaxes_history_without_crashing(self):
        """Simula biblioteca pequena onde todas as músicas estão no histórico, confirmando relaxamento e sem travamento."""
        original_query = mpd_integration.vector_store.query_similar
        try:
            mpd_integration.vector_store.query_similar = lambda col, emb, k=5, exclude_id=None: [
                {"track_id": "B", "distance": 0.05, "filename": "track_b.mp3"},
                {"track_id": "C", "distance": 0.10, "filename": "track_c.mp3"},
            ]

            # Histórico contém todas as candidatas possíveis: B e C
            history = deque(["B", "C"], maxlen=10)

            res = mpd_integration.select_next_recommendations(
                collection=self.collection,
                current_track_id="A",
                id_to_file_map=self.id_to_file,
                history=history,
                selection_mode="first"
            )

            # Não deve travar nem retornar None: deve relaxar o item mais antigo ('B') e retornar 'B'
            self.assertIsNotNone(res)
            self.assertEqual(res[0], "rock/track_b.mp3")

        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_mpd_stop_state_with_empty_queue(self):
        """Simula o MPD em estado 'stop' com fila vazia, confirmando que a fila é reposta sem lançar exceções."""
        original_query = mpd_integration.vector_store.query_similar
        try:
            mpd_integration.vector_store.query_similar = lambda col, emb, k=5, exclude_id=None: [
                {"track_id": "B", "distance": 0.05, "filename": "track_b.mp3"},
                {"track_id": "C", "distance": 0.10, "filename": "track_c.mp3"},
            ]

            mock_client = MagicMock()
            mock_client.status.return_value = {"song": "-1", "state": "stop"}
            mock_client.playlistinfo.side_effect = [
                [], # vazia
                [{"file": "rock/track_b.mp3"}],
                [{"file": "rock/track_b.mp3"}, {"file": "rock/track_c.mp3"}]
            ]

            history = deque(["A"], maxlen=10)
            failed = set()

            added = mpd_integration.replenish_mpd_queue(
                client=mock_client,
                collection=self.collection,
                current_track_id="A",
                file_to_id_map=self.file_to_id,
                id_to_file_map=self.id_to_file,
                music_root="",
                play_history=history,
                failed_tracks=failed,
                min_queue_ahead=2,
                max_tries=5
            )

            self.assertEqual(added, 2)
            self.assertEqual(mock_client.add.call_count, 2)

        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_session_history_atomic_save_and_corrupt_recovery(self):
        """Testa gravação atômica e recuperação graciosa caso session_history.json esteja corrompido."""
        test_file = "test_temp_session_history.json"
        try:
            if os.path.exists(test_file):
                os.remove(test_file)

            d = deque(["track_1", "track_2"], maxlen=10)
            mpd_integration.save_session_history(d, filepath=test_file)

            loaded = mpd_integration.load_session_history(filepath=test_file, max_len=10)
            self.assertEqual(list(loaded), ["track_1", "track_2"])

            # Corrompe o arquivo
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("CONTEUDO_INVALIDO_{{{")

            # Carregamento deve tolerar o arquivo corrompido e retornar deque vazio
            loaded_corrupt = mpd_integration.load_session_history(filepath=test_file, max_len=10)
            self.assertIsInstance(loaded_corrupt, deque)
            self.assertEqual(len(loaded_corrupt), 0)

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(f"{test_file}.tmp"):
                os.remove(f"{test_file}.tmp")

if __name__ == "__main__":
    unittest.main()
