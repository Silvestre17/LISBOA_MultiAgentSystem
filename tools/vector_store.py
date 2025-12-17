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
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

class KnowledgeBase:
    """
    Manages the RAG system using ChromaDB.
    """
    
    def __init__(self):
        # Initialize Embeddings (runs on CPU/GPU depending on torch setup)
        print(f"📥 Initializing Embeddings: {Config.EMBEDDING_MODEL_NAME}...")
        
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
            print(f"⚠️ Warning: File not found {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        documents = []
        for item in data:
            # Create a rich textual representation for embedding
            content = f"Name: {item.get('title', 'Unknown')}\n"
            content += f"Category: {item.get('category', 'General')}\n"
            content += f"Description: {item.get('full_description', '') or item.get('short_description', '')}\n"
            content += f"Location: {item.get('location', 'Lisbon')}\n"
            
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
            
        print(f"📖 Reading PDF: {file_path}")
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
            shutil.rmtree(self.vector_db_path)
            print("🗑️ Existing database removed.")

        # Check if DB exists
        if os.path.exists(self.vector_db_path) and not force_rebuild:
            print("✅ Database already exists. Skipping ingestion.")
            return

        print("🔄 Loading data sources...")
        
        # 1. Load JSONs
        docs_places = self._load_visitlisboa_json(str(Config.PATH_VISIT_LISBOA_PLACES), "VisitLisboa_Places")
        docs_events = self._load_visitlisboa_json(str(Config.PATH_VISIT_LISBOA_EVENTS), "VisitLisboa_Events")
        
        # 2. Load PDF directly
        docs_pdf = self._load_pdf_guide(str(Config.PATH_PDF_GUIDE))
        
        all_docs = docs_places + docs_events + docs_pdf
        
        if not all_docs:
            print("❌ No documents found to ingest!")
            return
        
        print(f"📊 Total documents to index: {len(all_docs)}")

        # Create/Update ChromaDB (Batch size limit prevents memory crash)
        Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path
        )
        
        print("🎉 Vector Store built successfully!")

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