import json
import os
from vector_store import get_client, get_or_create_collection

METADATA_PATH = "results/metadata.json"
MUSIC_ROOT = r"G:\Meu Drive\Musicas"

def clean():
    if not os.path.exists(METADATA_PATH):
        print("Metadata not found.")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    orphans = []
    keep = []
    
    for m in metadata:
        filepath = m.get('filepath', '')
        # Check if the file physically exists on the disk
        abs_path = os.path.join(MUSIC_ROOT, filepath)
        if not os.path.exists(abs_path):
            orphans.append(m)
        else:
            keep.append(m)

    if not orphans:
        print("No orphans found.")
        return

    print(f"Found {len(orphans)} orphans to delete.")
    for o in orphans:
        print(f"Orphan: {o.get('filepath')}")
    
    # ChromaDB
    client = get_client()
    collection = get_or_create_collection(client)
    
    orphan_ids = [m['track_id'] for m in orphans]
    print(f"Deleting IDs from Chroma: {orphan_ids}")
    
    try:
        collection.delete(ids=orphan_ids)
    except Exception as e:
        print(f"Error deleting from ChromaDB: {e}")
        
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=4)
        
    print(f"Kept {len(keep)} tracks.")

if __name__ == "__main__":
    clean()
