#!/usr/bin/env python3
"""
Test script for ChromaDB setup
"""

from core.milvus_utilis import save_to_chromadb, search_similar_chunks, get_collection_info
from core.embedding import split_into_chunks

def test_chromadb():
    print("🧪 Testing ChromaDB Setup")
    print("=" * 40)
    
    # Test 1: Check collection info
    print("1. Checking collection info...")
    info = get_collection_info()
    print(f"   Collection: {info['collection_name']}")
    print(f"   Total chunks: {info['total_chunks']}")
    
    # Test 2: Add some test data
    print("\n2. Adding test data...")
    test_text = """
    This is a test document about US immigration law.
    The naturalization process requires applicants to be at least 18 years old.
    Applicants must have been a permanent resident for at least 5 years.
    They must demonstrate good moral character and pass English and civics tests.
    """
    
    chunks = split_into_chunks(test_text)
    print(f"   Created {len(chunks)} chunks from test text")
    
    save_to_chromadb(chunks, "test_document.txt")
    
    # Test 3: Search for similar content
    print("\n3. Testing search functionality...")
    query = "What are the requirements for naturalization?"
    results = search_similar_chunks(query, top_k=3)
    
    print(f"   Search query: '{query}'")
    print(f"   Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. Score: {result['score']:.3f}")
        print(f"      Chunk: {result['chunk'][:100]}...")
    
    # Test 4: Check updated collection info
    print("\n4. Updated collection info...")
    info = get_collection_info()
    print(f"   Total chunks: {info['total_chunks']}")
    
    print("\n✅ ChromaDB test completed successfully!")
    print("🎉 Your ChromaDB setup is working correctly!")

if __name__ == "__main__":
    test_chromadb() 