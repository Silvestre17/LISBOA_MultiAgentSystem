# ==========================================================================
# Master Thesis - Vector Store Management
#   - André Filipe Gomes Silvestre, 2025
# 
#   Handles ingestion of static data (VisitLisboa JSONs + PDF Text) into ChromaDB.
#   Uses HuggingFace Embeddings.
# ==========================================================================

import os
import sys
import json
import shutil
import stat
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Disable telemetry for HuggingFace
# Telemetry mechanisms from the vector database backend were disabled, 
# as they are not relevant to system performance or retrieval quality.
os.environ["ANONYMIZED_TELEMETRY"] = "false"

class KnowledgeBase:
    """
    Manages the RAG system using ChromaDB.
    """
    
    def __init__(self):
        # Initialize Embeddings (runs on CPU/GPU depending on torch setup)
        print(f"\033[1m📥 Initializing Embeddings:\033[0m {Config.EMBEDDING_MODEL_NAME}...")
        
        # model_kwargs={'device': 'cuda'} forces GPU if available
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_db_path = str(Config.VECTOR_DB_DIR)
        
    def _load_visitlisboa_json(self, file_path: str, source_tag: str) -> List[Document]:
        """Converts VisitLisboa JSON structure into LangChain Documents."""
        if not os.path.exists(file_path):
            print(f"\033[1m⚠️ Warning:\033[0m File not found {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        documents = []
        for item in data:
            # Create a rich textual representation for embedding
            # Dynamically include all available fields
            content_parts = []
            
            # Prioritize Title for readability
            if 'title' in item:
                content_parts.append(f"Name: {item['title']}")
            
            for key, value in item.items():
                if key == 'title': continue  # Already added
                if not value: continue       # Skip empty fields
                
                # Format complex types (lists/dicts)
                if isinstance(value, list):
                    val_str = ", ".join(map(str, value))
                elif isinstance(value, dict):
                    val_str = json.dumps(value, ensure_ascii=False)
                else:
                    val_str = str(value)
                
                # Clean up key name (e.g. full_description -> Full Description)
                key_clean = key.replace('_', ' ').title()
                content_parts.append(f"{key_clean}: {val_str}")
            
            content = "\n".join(content_parts)
            
            # Store structured data in metadata for retrieval later
            meta = {
                "source": source_tag,
                "title": item.get('title', 'Unknown'),
                "url": item.get('url', ''),
                "category": item.get('category', 'General')
            }
            
            documents.append(Document(page_content=content, metadata=meta))
        
        return documents

    def _load_pdf_guide(self, file_path: str) -> List[Document]:
        """
        Loads the PDF guide directly using LangChain's PyPDFLoader.
        """
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: PDF File not found {file_path}")
            return []
            
        print(f"\033[1m📖 Reading PDF:\033[0m {file_path}")
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        
        # Split into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        docs = text_splitter.split_documents(pages)
        
        # Add extra metadata to indicate the source is the official guide PDF
        for doc in docs:
            doc.metadata["source"] = "TurismoLisboa_OfficialGuide_PDF"
            
        print(f"   -> PDF split into {len(docs)} chunks.")
        return docs

    def build_database(self, force_rebuild: bool = False):
        """Ingests all data into ChromaDB."""
        
        if force_rebuild and os.path.exists(self.vector_db_path):
            def on_rm_error(func, path, exc_info):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(self.vector_db_path, onerror=on_rm_error)
            print("\033[1m🗑️ Existing database removed.\033[0m")

        # Check if DB exists
        if os.path.exists(self.vector_db_path) and not force_rebuild:
            print("\033[1m✅ Database already exists. Skipping ingestion.\033[0m")
            return

        print("\033[1m🔄 Loading data sources...\033[0m")
        
        # 1. Load JSONs
        docs_places = self._load_visitlisboa_json(str(Config.PATH_VISIT_LISBOA_PLACES), "VisitLisboa_Places")
        docs_events = self._load_visitlisboa_json(str(Config.PATH_VISIT_LISBOA_EVENTS), "VisitLisboa_Events")
        
        # 2. Load PDF directly
        docs_pdf = self._load_pdf_guide(str(Config.PATH_PDF_TEXT))
        
        all_docs = docs_places + docs_events + docs_pdf
        
        if not all_docs:
            print("\033[1m❌ No documents found to ingest!\033[0m")
            return
        
        print(f"\033[1m📊 Total documents to index:\033[0m {len(all_docs)}")

        # Create/Update ChromaDB (Batch size limit prevents memory crash)
        Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path
        )
        
        print("\033[1m🎉 Vector Store built successfully!\033[0m")

    def get_retriever_tool(self):
        """Returns the vector store as a retriever for LangChain."""
        vectorstore = Chroma(
            persist_directory=self.vector_db_path, 
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})

# Script to run manually to build the DB
if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.build_database(force_rebuild=True)