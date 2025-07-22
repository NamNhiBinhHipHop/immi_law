#!/usr/bin/env python3
"""
Setup script for ChromaDB migration
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    print("🔧 Setting up ChromaDB for AI Document Assistant")
    print("=" * 50)
    
    # Check if chroma_db directory exists
    chroma_dir = Path("./chroma_db")
    if chroma_dir.exists():
        print("✅ ChromaDB directory already exists")
    else:
        print("📁 ChromaDB directory will be created on first run")
    
    # Check if old Milvus volumes exist
    volumes_dir = Path("./volumes")
    if volumes_dir.exists():
        print("⚠️  Found old Milvus volumes directory")
        response = input("Do you want to remove the old Milvus data? (y/N): ")
        if response.lower() == 'y':
            try:
                shutil.rmtree(volumes_dir)
                print("✅ Removed old Milvus volumes")
            except Exception as e:
                print(f"❌ Error removing volumes: {e}")
        else:
            print("ℹ️  Keeping old Milvus data")
    
    # Check if docker-compose.yml exists
    docker_compose = Path("./docker-compose.yml")
    if docker_compose.exists():
        print("⚠️  Found docker-compose.yml (no longer needed)")
        response = input("Do you want to remove it? (y/N): ")
        if response.lower() == 'y':
            try:
                docker_compose.unlink()
                print("✅ Removed docker-compose.yml")
            except Exception as e:
                print(f"❌ Error removing file: {e}")
        else:
            print("ℹ️  Keeping docker-compose.yml")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Configure your LLM API in config/config.py")
    print("3. Run the application: python cli_app.py --interactive")
print("   or: streamlit run streamlit_app.py")
print("\nNote: Only .txt and .md files are supported (PDF support removed)")

if __name__ == "__main__":
    main() 