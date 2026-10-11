import json
import os
from vector_store import get_client, get_or_create_collection

METADATA_PATH = "results/metadata.json"

def clean():
    if not os.path.exists(METADATA_PATH):
        print("Metadata not found.")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # find orphans
    orphans = [m for m in metadata if str(m.get('filepath', '')).upper().startswith('C:')]
    keep = [m for m in metadata if not str(m.get('filepath', '')).upper().startswith('C:')]

    if not orphans:
        print("No orphans found.")
        return

    print(f"Found {len(orphans)} orphans to delete.")
    
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
