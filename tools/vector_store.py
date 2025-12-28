# ==========================================================================
# Master Thesis - Vector Store Management
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Manages the RAG (Retrieval-Augmented Generation) knowledge base.
#   Handles ingestion of static data sources into ChromaDB for semantic search.
# 
#   Data Sources:
#     - VisitLisboa Places JSON: Museums, monuments, attractions
#     - VisitLisboa Events JSON: Cultural events, exhibitions
#     - Turismo de Lisboa PDF Guide: Official tourist information
# 
#   Features:
#     - HuggingFace embeddings (BAAI/bge-m3 for multilingual support)
#     - ChromaDB for persistent vector storage
#     - Automatic chunking with overlap for context preservation
#     - Metadata preservation for source attribution
# 
#   Usage:
#     # Build the database (run once)
#     python tools/vector_store.py
#     
#     # Use in code
#     from tools.vector_store import KnowledgeBase
#     kb = KnowledgeBase()
#     retriever = kb.get_retriever_tool()
# ==========================================================================

# Required libraries:
# pip install langchain-chroma langchain-huggingface pypdf langchain-community tqdm

import os
import sys
import json
import shutil
import stat
import warnings
from typing import List, Optional
from tqdm import tqdm

# Suppress opentelemetry warnings that may arise from version conflicts
warnings.filterwarnings("ignore", category=DeprecationWarning, module="opentelemetry")
os.environ["OTEL_SDK_DISABLED"] = "true"  # Disable OpenTelemetry SDK entirely

# Import ChromaDB with error handling for version conflicts
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
CHROMADB_AVAILABLE = True

# Add parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Disable telemetry for HuggingFace (privacy and performance)
os.environ["ANONYMIZED_TELEMETRY"] = "false"


class KnowledgeBase:
    """
    Manages the RAG (Retrieval-Augmented Generation) knowledge base.
    
    This class handles:
        1. Loading and processing static data sources
        2. Creating embeddings using HuggingFace models
        3. Storing vectors in ChromaDB for semantic search
        4. Providing retrieval functionality for the agent
    
    Attributes:
        embeddings (HuggingFaceEmbeddings): Embedding model instance.
        vector_db_path (str): Path to the ChromaDB persistence directory.
    
    Example:
        >>> kb = KnowledgeBase()
        >>> kb.build_database()  # First time only
        >>> retriever = kb.get_retriever_tool()
        >>> docs = retriever.invoke("museums in Lisbon")
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initializes the KnowledgeBase with embedding model.
        
        Args:
            use_gpu (bool): Whether to use GPU for embeddings (requires CUDA).
                           Falls back to CPU if GPU is unavailable.
        
        Raises:
            RuntimeError: If ChromaDB dependencies are not available.
        
        Notes:
            The embedding model (BAAI/bge-m3) is multilingual and works
            well with both Portuguese and English text.
        """
        # Check if dependencies are available
        if not CHROMADB_AVAILABLE:
            raise RuntimeError(
                "ChromaDB dependencies not available. "
                "Install with: pip install langchain-chroma langchain-huggingface"
            )
        
        print(f"\033[1m📥 Initializing Embeddings:\033[0m {Config.EMBEDDING_MODEL_NAME}...")
        
        # Configure device (GPU if available and requested)
        device = 'cuda' if use_gpu else 'cpu'
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
            )
            print(f"   \033[1;32m✓ Running on {device.upper()}\033[0m")
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"   \033[1;33m⚠ GPU unavailable, falling back to CPU\033[0m")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # Path for persistent ChromaDB storage
        self.vector_db_path = str(Config.VECTOR_DB_DIR)
        
    def _load_visitlisboa_json(self, file_path: str, source_tag: str) -> List[Document]:
        """
        Loads and processes VisitLisboa JSON data into LangChain Documents.
        
        Args:
            file_path (str): Path to the JSON file.
            source_tag (str): Tag to identify the source in metadata.
        
        Returns:
            List[Document]: List of LangChain Document objects.
        
        Notes:
            Each item in the JSON is converted to a rich text representation
            suitable for embedding, with metadata preserved for retrieval.
        """
        if not os.path.exists(file_path):
            print(f"\033[1;33m⚠️ Warning:\033[0m File not found: {file_path}")
            return []

        print(f"\033[1m📂 Loading:\033[0m {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        documents = []
        for item in data:
            # Build a rich text representation for embedding
            # This format helps the model understand the content structure
            content_parts = []
            
            # Title is prioritized for better search matching
            if 'title' in item:
                content_parts.append(f"Name: {item['title']}")
            
            # Process all other fields
            for key, value in item.items():
                if key == 'title':
                    continue  # Already added
                if not value:
                    continue  # Skip empty fields
                
                # Format complex types (lists, dicts)
                if isinstance(value, list):
                    val_str = ", ".join(map(str, value))
                elif isinstance(value, dict):
                    val_str = json.dumps(value, ensure_ascii=False)
                else:
                    val_str = str(value)
                
                # Clean up key name (e.g., full_description -> Full Description)
                key_clean = key.replace('_', ' ').title()
                content_parts.append(f"{key_clean}: {val_str}")
            
            content = "\n".join(content_parts)
            
            # Metadata for source attribution and filtering
            meta = {
                "source": source_tag,
                "title": item.get('title', 'Unknown'),
                "url": item.get('url', ''),
                "category": item.get('category', 'General')
            }
            
            documents.append(Document(page_content=content, metadata=meta))
        
        print(f"   \033[1;32m✓ Loaded {len(documents)} items\033[0m")
        return documents

    def _load_pdf_guide(self, file_path: str) -> List[Document]:
        """
        Loads and chunks the PDF tourist guide using PyPDFLoader.
        
        Args:
            file_path (str): Path to the PDF file.
        
        Returns:
            List[Document]: List of chunked Document objects.
        
        Notes:
            The PDF is split into chunks of ~1000 characters with 200 char
            overlap to preserve context across chunk boundaries.
        """
        if not os.path.exists(file_path):
            print(f"\033[1;33m⚠️ Warning:\033[0m PDF not found: {file_path}")
            return []
            
        print(f"\033[1m📖 Loading PDF:\033[0m {os.path.basename(file_path)}")
        
        # Load PDF pages
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        
        # Configure text splitter for optimal chunk size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,           # Target chunk size in characters
            chunk_overlap=200,         # Overlap to preserve context
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Split into chunks
        docs = text_splitter.split_documents(pages)
        
        # Add source metadata to all chunks
        for doc in docs:
            doc.metadata["source"] = "TurismoLisboa_OfficialGuide_PDF"
            
        print(f"   \033[1;32m✓ Split into {len(docs)} chunks\033[0m")
        return docs

    def build_database(self, force_rebuild: bool = False) -> bool:
        """
        Builds or rebuilds the ChromaDB vector database.
        
        Args:
            force_rebuild (bool): If True, deletes existing database and rebuilds.
                                 If False, skips if database exists.
        
        Returns:
            bool: True if database was built successfully.
        
        Notes:
            This method should be run once initially, then only when
            source data is updated. Building takes several minutes
            depending on data size and hardware.
        """
        # Handle force rebuild
        if force_rebuild and os.path.exists(self.vector_db_path):
            def on_rm_error(func, path, exc_info):
                """Error handler for permission issues on Windows."""
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(self.vector_db_path, onerror=on_rm_error)
            print("\033[1m🗑️ Existing database removed.\033[0m")

        # Skip if database exists and not forcing rebuild
        if os.path.exists(self.vector_db_path) and not force_rebuild:
            print("\033[1;32m✅ Database already exists. Use force_rebuild=True to rebuild.\033[0m")
            return True

        print("\033[1m" + "=" * 50 + "\033[0m")
        print("\033[1m🔄 Building Vector Database\033[0m")
        print("\033[1m" + "=" * 50 + "\033[0m")
        
        # Load all data sources
        docs_places = self._load_visitlisboa_json(
            str(Config.PATH_VISIT_LISBOA_PLACES), 
            "VisitLisboa_Places"
        )
        docs_events = self._load_visitlisboa_json(
            str(Config.PATH_VISIT_LISBOA_EVENTS), 
            "VisitLisboa_Events"
        )
        docs_pdf = self._load_pdf_guide(str(Config.PATH_PDF_TEXT))
        
        # Combine all documents
        all_docs = docs_places + docs_events + docs_pdf
        
        if not all_docs:
            print("\033[1;31m❌ No documents found to ingest!\033[0m")
            return False
        
        print(f"\n\033[1m📊 Total documents to index:\033[0m {len(all_docs)}")
        print("   Creating embeddings and storing in ChromaDB...\n")

        # Process documents in batches with progress tracking
        # Batch size of 100 balances progress visibility with performance
        batch_size = 100
        vectorstore = None
        
        # Create progress bar
        pbar = tqdm(
            total=len(all_docs),
            desc="   🔄 Indexing",
            unit="docs",
            bar_format="   {desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i + batch_size]
            
            if vectorstore is None:
                # First batch: create new ChromaDB
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.vector_db_path
                )
            else:
                # Subsequent batches: add to existing
                vectorstore.add_documents(batch)
            
            pbar.update(len(batch))
        
        pbar.close()
        
        print("\n\033[1;32m🎉 Vector Store built successfully!\033[0m")
        print(f"   Location: {self.vector_db_path}")
        return True

    def get_retriever(self, k: int = 5):
        """
        Returns a retriever interface for semantic search.
        
        Args:
            k (int): Number of documents to retrieve per query.
        
        Returns:
            VectorStoreRetriever: Configured retriever for use with LangChain.
        
        Example:
            >>> retriever = kb.get_retriever(k=3)
            >>> docs = retriever.invoke("best museums in Lisbon")
        """
        vectorstore = Chroma(
            persist_directory=self.vector_db_path, 
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Performs a semantic search on the knowledge base.
        
        Args:
            query (str): Search query in natural language.
            k (int): Number of results to return.
        
        Returns:
            List[Document]: Most relevant documents.
        
        Example:
            >>> results = kb.search("museums near Belém")
            >>> for doc in results:
            ...     print(doc.metadata['title'])
        """
        retriever = self.get_retriever(k=k)
        return retriever.invoke(query)

    def get_stats(self) -> dict:
        """
        Returns statistics about the vector database.
        
        Returns:
            dict: Statistics including document count and sources.
        """
        if not os.path.exists(self.vector_db_path):
            return {"status": "not_built", "count": 0}
        
        vectorstore = Chroma(
            persist_directory=self.vector_db_path, 
            embedding_function=self.embeddings
        )
        
        # Get collection stats
        collection = vectorstore._collection
        count = collection.count()
        
        return {
            "status": "ready",
            "count": count,
            "path": self.vector_db_path
        }


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    """
    Script to build or test the vector database.
    
    Usage:
        python tools/vector_store.py           # Build (skip if exists)
        python tools/vector_store.py --force   # Force rebuild
        python tools/vector_store.py --test    # Test search only
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Store Management")
    parser.add_argument("--force", action="store_true", help="Force rebuild database")
    parser.add_argument("--test", action="store_true", help="Test search only (skip build)")
    args = parser.parse_args()
    
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Vector Store Management\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    kb = KnowledgeBase()
    
    if args.test:
        # Test mode: search only
        print("\n\033[1m🔍 Testing Search...\033[0m")
        stats = kb.get_stats()
        print(f"   Database status: {stats['status']}")
        print(f"   Document count: {stats['count']}")
        
        if stats['status'] == 'ready':
            test_queries = [
                "museums in Belém",
                "restaurants with traditional food",
                "events this week"
            ]
            
            for query in test_queries:
                print(f"\n\033[1m📝 Query:\033[0m {query}")
                results = kb.search(query, k=3)
                for i, doc in enumerate(results, 1):
                    title = doc.metadata.get('title', 'N/A')[:50]
                    source = doc.metadata.get('source', 'N/A')
                    print(f"   {i}. {title} ({source})")
    else:
        # Build mode
        kb.build_database(force_rebuild=args.force)
        
        # Show stats
        stats = kb.get_stats()
        print(f"\n\033[1m📊 Database Stats:\033[0m")
        print(f"   Status: {stats['status']}")
        print(f"   Documents: {stats['count']}")