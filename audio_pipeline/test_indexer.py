import logging
from auto_indexer import index_unknown_track, load_known_fingerprints
from vector_store import get_or_create_collection
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

metadata_path = "./results/metadata.json"
music_root = "C:/mpd/music"
test_file = "C:/mpd/music/Century - Lover Why (Remix).mp3"

known_fp = load_known_fingerprints(metadata_path)
print(f"Loaded {len(known_fp)} known fingerprints")

import chromadb
client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
collection = get_or_create_collection(client, config.COLLECTION_NAME)

new_id = index_unknown_track(
    filepath=test_file,
    metadata_path=metadata_path,
    collection=collection,
    known_fingerprints=known_fp,
    music_root=music_root
)

print(f"Resulting Track ID: {new_id}")
