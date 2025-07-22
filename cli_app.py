#!/usr/bin/env python3
"""
AI Document Assistant - Command Line Interface
"""

import os
import sys
import argparse
import shlex
import json
import datetime
import re
from pathlib import Path
from typing import List, Dict, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.rag_chain import deep_search_pipeline
from core.milvus_utilis import save_to_chromadb, search_similar_chunks, delete_file, delete_all, collection
from core.embedding import split_into_chunks

class ConversationMemory:
    """Manages conversation history for the CLI app"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history: List[Dict] = []
        self.history_file = Path(f"conversation_history_{self.session_id}.json")
        
    def _clean_answer(self, answer: str) -> str:
        """Remove thinking tags and clean up the answer"""
        # Remove <think>...</think> tags and their content
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        # Remove any remaining thinking-related text
        answer = re.sub(r'<THINK>.*?</THINK>', '', answer, flags=re.DOTALL)
        # Clean up extra whitespace
        answer = re.sub(r'\n\s*\n', '\n\n', answer)
        answer = answer.strip()
        return answer
        
    def add_ask_query(self, question: str, answer: str):
        """Add an ask query with its answer"""
        # Clean the answer before storing
        cleaned_answer = self._clean_answer(answer)
        entry = {
            "question": question,
            "answer": cleaned_answer
        }
        self.history.append(entry)
        self._save_history()
        
    def show_history(self):
        """Display conversation history"""
        if not self.history:
            print("📝 No conversation history yet.")
            return
            
        print(f"\n📝 Conversation History:")
        print("=" * 50)
        
        for i, entry in enumerate(self.history, 1):
            question = entry["question"]
            answer = entry["answer"]
            
            print(f"\n{i}. Q: {question}")
            answer_preview = answer[:150] + "..." if len(answer) > 150 else answer
            print(f"   A: {answer_preview}")
        
        print("=" * 50)
        
    def _save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Warning: Could not save conversation history: {e}")
            
    def clear_history(self):
        """Clear conversation history and delete file"""
        self.history = []
        try:
            if self.history_file.exists():
                self.history_file.unlink()
                print("🗑️ Conversation history cleared.")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete history file: {e}")
            
    def get_context_summary(self) -> str:
        """Get a summary of recent conversation for context"""
        if not self.history:
            return ""
            
        recent_queries = self.history[-3:]  # Last 3 Q&A pairs
        summary_parts = []
        
        for i, entry in enumerate(recent_queries, 1):
            summary_parts.append(f"Q{i}: {entry['question']}")
            summary_parts.append(f"A{i}: {entry['answer']}")
                
        return "\n".join(summary_parts)

def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF support removed - this function is deprecated."""
    print(f"❌ PDF support has been removed. Cannot process {pdf_path}")
    return ""

def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading text file {txt_path}: {e}")
        return ""

def process_document(file_path: str) -> bool:
    """Process a document and add it to the vector database."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    # Check file size (skip files larger than 50MB)
    if file_path.stat().st_size > 50 * 1024 * 1024:
        print(f"⚠️ File too large ({file_path.stat().st_size / (1024*1024):.1f}MB), skipping...")
        return False
    
    # Extract text based on file type
    if file_path.suffix.lower() in ['.txt', '.md']:
        text = extract_text_from_txt(str(file_path))
    else:
        print(f"❌ Unsupported file type: {file_path.suffix}. Only .txt and .md files are supported.")
        return False
    
    if not text.strip():
        print(f"❌ No text extracted from {file_path}")
        return False
    
    # Split into chunks
    chunks = split_into_chunks(text)
    print(f"📄 Extracted {len(chunks)} chunks from {file_path.name}")
    
    # Save to ChromaDB
    try:
        save_to_chromadb(chunks, file_path.name)
        print(f"✅ Successfully processed {file_path.name}")
        return True
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        return False

def interactive_mode():
    """Run the assistant in interactive mode."""
    print("🤖 AI Document Assistant - Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  ask <question>     - Ask a question about your documents")
    print("  upload <file>      - Upload and process a document")
    print("  search <query>     - Search for similar content")
    print("  delete <filename>  - Delete a document from the database")
    print("  delete-all         - Delete ALL data from the database")
    print("  list               - List all documents in the database")
    print("  history            - Show conversation history")
    print("  help               - Show this help message")
    print("  quit               - Exit the application")
    print("=" * 50)
    print("💡 Tip: Use quotes for file paths with spaces, e.g., upload \"testing files/document.pdf\"")
    print("=" * 50)
    
    conversation_memory = ConversationMemory()
    
    while True:
        try:
            user_input = input("\n🤖 Assistant> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                print("👋 Goodbye!")
                conversation_memory.clear_history()
                break
                
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  ask <question>     - Ask a question about your documents")
                print("  upload <file>      - Upload and process a document")
                print("  search <query>     - Search for similar content")
                print("  delete <filename>  - Delete a document from the database")
                print("  delete-all         - Delete ALL data from the database")
                print("  list               - List all documents in the database")
                print("  history            - Show conversation history")
                print("  help               - Show this help message")
                print("  quit               - Exit the application")
                print("\n💡 Tip: Use quotes for file paths with spaces, e.g., upload \"testing files/document.pdf\"")
                
            elif user_input.lower() == 'history':
                conversation_memory.show_history()
                
            elif user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                if question:
                    print(f"\n🤔 Question: {question}")
                    print("🔄 Thinking...")
                    try:
                        # Get conversation context
                        context = conversation_memory.get_context_summary()
                        # Pass the context directly to the function
                        answer = deep_search_pipeline(question, chat_history=context)
                        conversation_memory.add_ask_query(question, answer)
                        print(f"\n💡 Answer: {answer}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Please provide a question after 'ask'")
                    
            elif user_input.lower().startswith('upload '):
                # Parse the upload command properly to handle spaces and quotes
                try:
                    parts = shlex.split(user_input)
                    if len(parts) >= 2:
                        file_path = parts[1]
                        print(f"📤 Uploading {file_path}...")
                        process_document(file_path)
                    else:
                        print("❌ Please provide a file path after 'upload'")
                except Exception as e:
                    print(f"❌ Error parsing file path: {e}")
                    print("💡 Tip: Use quotes for file paths with spaces, e.g., upload \"testing files/document.pdf\"")
                    
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    print(f"🔍 Searching for: {query}")
                    try:
                        results = search_similar_chunks(query, top_k=5)
                        print(f"\n📋 Found {len(results)} results:")
                        for i, result in enumerate(results, 1):
                            print(f"\n{i}. Score: {result['score']:.3f}")
                            print(f"   Content: {result['chunk'][:200]}...")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Please provide a search query after 'search'")
                    
            elif user_input.lower().startswith('delete '):
                filename = user_input[7:].strip()
                if filename:
                    print(f"🗑️ Deleting {filename}...")
                    try:
                        result = delete_file(filename)
                        print(f"✅ {result['message']}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Please provide a filename after 'delete'")
                    
            elif user_input.lower() == 'delete-all':
                print("🗑️ WARNING: This will delete ALL data from the database!")
                confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
                if confirm == 'yes':
                    try:
                        result = delete_all()
                        print(f"✅ {result['message']}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Delete all operation cancelled.")
                    
            elif user_input.lower() == 'list':
                try:
                    collection.load()
                    results = collection.query(
                        expr="",
                        output_fields=["filename"],
                        limit=1000
                    )
                    filenames = list(set([r["filename"] for r in results]))
                    if filenames:
                        print(f"\n📚 Documents in database ({len(filenames)}):")
                        for filename in filenames:
                            print(f"  - {filename}")
                    else:
                        print("📚 No documents in database")
                except Exception as e:
                    print(f"❌ Error listing documents: {e}")
                    
            else:
                # If no command is recognized, treat it as a question
                print(f"🤔 Question: {user_input}")
                print("🔄 Thinking...")
                try:
                    # Get conversation context
                    context = conversation_memory.get_context_summary()
                    # Pass the context directly to the function
                    answer = deep_search_pipeline(user_input, chat_history=context)
                    conversation_memory.add_ask_query(user_input, answer)
                    print(f"\n💡 Answer: {answer}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            conversation_memory.clear_history()
            break
        except EOFError:
            print("\n👋 Goodbye!")
            conversation_memory.clear_history()
            break

def main():
    parser = argparse.ArgumentParser(description="AI Document Assistant CLI")
    parser.add_argument("--ask", "-a", help="Ask a question directly")
    parser.add_argument("--upload", "-u", help="Upload and process a document")
    parser.add_argument("--search", "-s", help="Search for similar content")
    parser.add_argument("--delete", "-d", help="Delete a document from the database")
    parser.add_argument("--delete-all", action="store_true", help="Delete ALL data from the database")
    parser.add_argument("--list", "-l", action="store_true", help="List all documents")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not any([args.ask, args.upload, args.search, args.delete, args.delete_all, args.list]):
        interactive_mode()
    else:
        # Single command mode
        if args.ask:
            print(f"🤔 Question: {args.ask}")
            print("🔄 Thinking...")
            try:
                answer = deep_search_pipeline(args.ask, chat_history="")
                print(f"\n💡 Answer: {answer}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif args.upload:
            print(f"📤 Uploading {args.upload}...")
            process_document(args.upload)
            
        elif args.search:
            print(f"🔍 Searching for: {args.search}")
            try:
                results = search_similar_chunks(args.search, top_k=5)
                print(f"\n📋 Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Score: {result['score']:.3f}")
                    print(f"   Content: {result['chunk'][:200]}...")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif args.delete:
            print(f"🗑️ Deleting {args.delete}...")
            try:
                result = delete_file(args.delete)
                print(f"✅ {result['message']}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif args.delete_all:
            print("🗑️ WARNING: This will delete ALL data from the database!")
            confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
            if confirm == 'yes':
                try:
                    result = delete_all()
                    print(f"✅ {result['message']}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ Delete all operation cancelled.")
                
        elif args.list:
            try:
                collection.load()
                results = collection.query(
                    expr="",
                    output_fields=["filename"],
                    limit=1000
                )
                filenames = list(set([r["filename"] for r in results]))
                if filenames:
                    print(f"\n📚 Documents in database ({len(filenames)}):")
                    for filename in filenames:
                        print(f"  - {filename}")
                else:
                    print("📚 No documents in database")
            except Exception as e:
                print(f"❌ Error listing documents: {e}")

if __name__ == "__main__":
    main() 
