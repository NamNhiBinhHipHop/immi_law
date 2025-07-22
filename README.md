# 🇺🇸 AI Immigration Law Assistant

A powerful Retrieval-Augmented Generation (RAG) system for answering US immigration and citizenship questions using official law documents. Built with Python, ChromaDB, and modern NLP models.

---

## 🛠️ Quickstart

### 1. Clone & Install
```bash
git clone <repo-url>
cd chatbot-main
python3 -m venv env311
source env311/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Your Data
- Place your immigration law `.txt` or `.md` files in the project directory.
- Example: `full_immigration_law.txt`

### 3. Run Setup (Optional)
```bash
python setup_chromadb.py
```

### 4. Upload a Document
```bash
python cli_app.py --upload "full_immigration_law.txt"
```

### 5. Start the Web App
```bash
streamlit run streamlit_app.py
```
- Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Configuration
- **Vector DB**: ChromaDB (persistent, local, no Docker)
- **Chunking**: `CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 200`
- **Supported Files**: `.txt`, `.md` only (no PDF)
- **Embeddings**: `sentence-transformers` (MiniLM-L6-v2)

---

## 📝 Example Usage
- **Ask a question:**
  - "What are the requirements for naturalization?"
  - "How do I apply for a green card renewal?"
- **Upload new law text:**
  - `python cli_app.py --upload "uscis_policy_2024.txt"`

---

## 🧹 Maintenance
- **Delete a file from DB:**
  - `python cli_app.py --delete "filename.txt"`
- **Clear all data:**
  - `python cli_app.py --delete-all`
- **ChromaDB data directory:**
  - `chroma_db/` (auto-created, add to `.gitignore`)

---

## 🐍 Requirements
- Python 3.11+
- No GPU required
- 4GB+ RAM recommended

---

## 🙏 Acknowledgments
- [ChromaDB](https://www.trychroma.com/) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [USCIS](https://www.uscis.gov/) for official immigration law

---

## 📢 Notes
- **No PDF support**: Only plain text and markdown files are supported.
- **Chunking**: Large documents are split into overlapping 1000-character chunks for optimal search.
- **Privacy**: All data stays local unless you connect to an external LLM API.

---

## 💬 Need Help?
Open an issue or PR, or contact the maintainer for support.
