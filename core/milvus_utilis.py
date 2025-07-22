import chromadb
from chromadb.config import Settings
from core.embedding import embed_chunks, split_into_chunks
import uuid
import time
import os
from typing import List, Dict, Optional

print("🕒 Initializing ChromaDB...")
start_time = time.time()

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Get or create collection
try:
    collection = chroma_client.get_collection("documents")
    print("✅ Connected to existing ChromaDB collection")
except:
    collection = chroma_client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    print("✅ Created new ChromaDB collection")

connect_time = time.time() - start_time
print(f"✅ ChromaDB initialized in {connect_time:.2f} seconds")

def save_to_chromadb(chunks: List[str], filename: str, vectors: Optional[List[List[float]]] = None):
    """
    Save text chunks and their vectors to ChromaDB. Each chunk gets a unique ID.
    Handles large documents by processing in batches.
    """
    start_time = time.time()
    
    # Generate embeddings if not provided
    if vectors is None:
        print(f"⏱️ Embedding {len(chunks)} chunks...")
        vectors = embed_chunks(chunks)
        print(f"⏱️ Embedding completed in {time.time() - start_time:.2f} seconds")
    
    # Process in batches to avoid ChromaDB limits
    batch_size = 10000  # Safe batch size for ChromaDB
    total_saved = 0
    
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end_idx]
        batch_vectors = vectors[i:end_idx]
        
        # Generate unique IDs for this batch
        batch_ids = [str(uuid.uuid4()) for _ in batch_chunks]
        
        # Prepare metadata for this batch
        batch_metadatas = [{"filename": filename, "chunk_index": j} for j in range(i, end_idx)]

        # Insert batch into ChromaDB
        collection.add(
            ids=batch_ids,
            embeddings=batch_vectors,
            documents=batch_chunks,
            metadatas=batch_metadatas
        )
        
        total_saved += len(batch_chunks)
        print(f"📦 Saved batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")
    
    save_time = time.time() - start_time
    print(f"✅ Successfully saved {total_saved} chunks to ChromaDB in {save_time:.2f} seconds")

def search_similar_chunks(query: str, top_k: int = 1000) -> List[Dict]:
    """
    Embed the query and find the top_k most similar text chunks in ChromaDB.
    """
    start_time = time.time()
    
    # Get query embedding
    query_vectors = embed_chunks([query])
    if not query_vectors:
        raise ValueError("Failed to generate embedding for query")
    
    query_vector = query_vectors[0]  # Use the first (and only) vector

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    matches = []
    if results['documents'] and results['documents'][0]:
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            # Convert distance to similarity score (ChromaDB uses cosine distance)
            # Cosine distance ranges from 0 to 2, where 0 is most similar
            similarity_score = 1 - (distance / 2)  # Convert to 0-1 range where 1 is most similar
            matches.append({
                "score": similarity_score,
                "chunk": doc,
                "filename": metadata.get("filename", "unknown")
            })

    search_time = time.time() - start_time
    print(f"⏱️ Search completed in {search_time:.2f} seconds")
    return matches
    
def delete_file(filename: str) -> Dict:
    """
    Delete all chunks of a file from ChromaDB by filename.
    """
    try:
        # Get all documents with the specified filename
        results = collection.get(
            where={"filename": filename},
            include=["ids"]
        )
        
        if results['ids']:
            # Delete by IDs
            collection.delete(ids=results['ids'])
            print(f"✅ Deleted {len(results['ids'])} chunks of {filename} from ChromaDB.")
            return {
                "filename": filename,
                "message": f"✅ Deleted {len(results['ids'])} chunks of {filename} from ChromaDB."
            }
        else:
            print(f"📭 No chunks found for {filename}")
            return {
                "filename": filename,
                "message": f"📭 No chunks found for {filename}"
            }
    except Exception as e:
        print(f"❌ Error deleting file {filename}: {e}")
        return {
            "filename": filename,
            "message": f"❌ Error deleting file {filename}: {e}"
        }

def delete_all() -> Dict:
    """
    Delete all data from the ChromaDB collection.
    """
    try:
        # Get all documents
        results = collection.get(
            include=["ids"]
        )
        
        if not results['ids']:
            print("📭 No data to delete - collection is already empty.")
            return {
                "message": "📭 No data to delete - collection is already empty."
            }
        
        # Delete all documents
        collection.delete(ids=results['ids'])
        
        print(f"🗑️ Deleted {len(results['ids'])} records from ChromaDB collection.")
        
        return {
            "message": f"✅ Successfully deleted {len(results['ids'])} records from the database."
        }
    except Exception as e:
        print(f"❌ Error deleting all data: {e}")
        return {
            "message": f"❌ Error deleting all data: {e}"
        }

def get_collection_info() -> Dict:
    """
    Get information about the collection.
    """
    try:
        count = collection.count()
        return {
            "total_chunks": count,
            "collection_name": "documents"
        }
    except Exception as e:
        return {
            "error": f"Could not get collection info: {e}"
        }

# For backward compatibility, keep the collection reference
# but it's not used in the new implementation
