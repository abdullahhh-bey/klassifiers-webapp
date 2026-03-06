import chromadb
from pprint import pprint
import sys

# Change this to the connection ID you want to inspect
connection_id = 3
persist_directory = f"./chroma_db_connections/connection_{connection_id}"

try:
    # Initialize the Chroma Client pointing to the directory
    client = chromadb.PersistentClient(path=persist_directory)

    # List all collections in this Chroma DB
    collections = client.list_collections()
    if not collections:
        print("No collections found in this database.")
        sys.exit(0)
    
    # We typically just use the default collection or the one listed
    print(f"Collections found: {[c.name for c in collections]}")
    
    for coll in collections:
        collection = client.get_collection(name=coll.name)
        
        # Get all items in the collection
        # We fetch up to 100 items for display
        data = collection.get(limit=100)
        
        print(f"\n--- Data in Collection: '{coll.name}' ---")
        print(f"Total documents: {collection.count()}")
        print("\nDocument Snippets:")
        for idx, (doc_id, doc, metadata) in enumerate(zip(data['ids'], data['documents'], data['metadatas'])):
            print(f"\n[ID: {doc_id}]")
            if metadata:
                print(f"Metadata: {metadata}")
            print(f"Content:\n{doc}")
            
except Exception as e:
    print(f"Error reading ChromaDB: {e}")
