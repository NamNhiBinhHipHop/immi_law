#!/usr/bin/env python3
"""
Test script to find optimal chunk size and overlap for immigration law document
"""

import time
from core.embedding import split_into_chunks
from core.milvus_utilis import search_similar_chunks

def test_chunk_configurations():
    """Test different chunk configurations and measure performance"""
    
    # Read a sample of the document
    print("📖 Reading immigration law document sample...")
    with open("full_immigration_law.txt", "r", encoding="utf-8") as f:
        text = f.read(50000)  # Read first 50KB for testing
    
    print(f"📄 Sample text length: {len(text)} characters")
    
    # Test configurations
    configurations = [
        (200, 20),   # Small chunks, small overlap
        (300, 50),   # Current configuration
        (400, 80),   # Larger chunks, more overlap
        (500, 100),  # Even larger chunks
        (600, 120),  # Large chunks
        (800, 160),  # Very large chunks
        (1000, 200), # Extra large chunks
    ]
    
    results = []
    
    for chunk_size, overlap in configurations:
        print(f"\n🧪 Testing: chunk_size={chunk_size}, overlap={overlap}")
        
        # Update the configuration temporarily
        import core.embedding
        original_chunk_size = core.embedding.CHUNK_SIZE
        original_overlap = core.embedding.CHUNK_OVERLAP
        
        core.embedding.CHUNK_SIZE = chunk_size
        core.embedding.CHUNK_OVERLAP = overlap
        
        # Time the chunking process
        start_time = time.time()
        chunks = split_into_chunks(text)
        chunking_time = time.time() - start_time
        
        print(f"   📊 Generated {len(chunks)} chunks in {chunking_time:.2f}s")
        
        # Test search performance with a sample query
        test_query = "What are the requirements for naturalization?"
        
        start_time = time.time()
        results_search = search_similar_chunks(test_query, top_k=5)
        search_time = time.time() - start_time
        
        # Analyze chunk quality
        avg_chunk_length = sum(len(chunk) for chunk in chunks) / len(chunks)
        min_chunk_length = min(len(chunk) for chunk in chunks)
        max_chunk_length = max(len(chunk) for chunk in chunks)
        
        # Check if chunks contain complete sentences
        complete_sentences = sum(1 for chunk in chunks if chunk.strip().endswith(('.', '!', '?')))
        sentence_completeness = complete_sentences / len(chunks) * 100
        
        result = {
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_chunks': len(chunks),
            'chunking_time': chunking_time,
            'search_time': search_time,
            'avg_chunk_length': avg_chunk_length,
            'min_chunk_length': min_chunk_length,
            'max_chunk_length': max_chunk_length,
            'sentence_completeness': sentence_completeness,
            'top_result_score': results_search[0]['score'] if results_search else 0
        }
        
        results.append(result)
        
        print(f"   ⏱️  Chunking: {chunking_time:.2f}s, Search: {search_time:.2f}s")
        print(f"   📏 Avg chunk length: {avg_chunk_length:.0f} chars")
        print(f"   📝 Sentence completeness: {sentence_completeness:.1f}%")
        print(f"   🎯 Top result score: {result['top_result_score']:.3f}")
        
        # Restore original configuration
        core.embedding.CHUNK_SIZE = original_chunk_size
        core.embedding.CHUNK_OVERLAP = original_overlap
    
    # Find optimal configuration
    print(f"\n🎯 ANALYSIS RESULTS:")
    print("=" * 80)
    
    # Sort by different criteria
    by_search_score = sorted(results, key=lambda x: x['top_result_score'], reverse=True)
    by_chunking_speed = sorted(results, key=lambda x: x['chunking_time'])
    by_sentence_completeness = sorted(results, key=lambda x: x['sentence_completeness'], reverse=True)
    
    print(f"🏆 Best search relevance: {by_search_score[0]['chunk_size']}/{by_search_score[0]['overlap']} (score: {by_search_score[0]['top_result_score']:.3f})")
    print(f"⚡ Fastest chunking: {by_chunking_speed[0]['chunk_size']}/{by_chunking_speed[0]['overlap']} ({by_chunking_speed[0]['chunking_time']:.2f}s)")
    print(f"📝 Best sentence completeness: {by_sentence_completeness[0]['chunk_size']}/{by_sentence_completeness[0]['overlap']} ({by_sentence_completeness[0]['sentence_completeness']:.1f}%)")
    
    # Calculate balanced score (considering multiple factors)
    for result in results:
        # Normalize scores (0-1 range)
        norm_search_score = result['top_result_score']
        norm_chunking_speed = 1 - (result['chunking_time'] / max(r['chunking_time'] for r in results))
        norm_sentence_completeness = result['sentence_completeness'] / 100
        
        # Weighted score (prioritize search relevance and sentence completeness)
        balanced_score = (norm_search_score * 0.5 + norm_sentence_completeness * 0.3 + norm_chunking_speed * 0.2)
        result['balanced_score'] = balanced_score
    
    best_balanced = max(results, key=lambda x: x['balanced_score'])
    print(f"🎯 Best balanced configuration: {best_balanced['chunk_size']}/{best_balanced['overlap']} (score: {best_balanced['balanced_score']:.3f})")
    
    # Detailed recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 80)
    
    if best_balanced['chunk_size'] != 300 or best_balanced['overlap'] != 50:
        print(f"🔄 Consider updating current settings ({300}/{50}) to {best_balanced['chunk_size']}/{best_balanced['overlap']}")
        print(f"   Expected improvements:")
        print(f"   - Search relevance: {best_balanced['top_result_score']:.3f} vs current")
        print(f"   - Sentence completeness: {best_balanced['sentence_completeness']:.1f}%")
        print(f"   - Processing time: {best_balanced['chunking_time']:.2f}s")
    else:
        print(f"✅ Current configuration ({300}/{50}) appears optimal!")
    
    # Show top 3 configurations
    print(f"\n🏅 TOP 3 CONFIGURATIONS:")
    top_3 = sorted(results, key=lambda x: x['balanced_score'], reverse=True)[:3]
    for i, config in enumerate(top_3, 1):
        print(f"{i}. {config['chunk_size']}/{config['overlap']} - Score: {config['balanced_score']:.3f}")
        print(f"   Search: {config['top_result_score']:.3f}, Sentences: {config['sentence_completeness']:.1f}%, Speed: {config['chunking_time']:.2f}s")

if __name__ == "__main__":
    test_chunk_configurations() 