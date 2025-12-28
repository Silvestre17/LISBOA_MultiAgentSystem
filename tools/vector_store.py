# ==========================================================================
# Master Thesis - Vector Store Management (Incremental Sync)
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Manages the RAG (Retrieval-Augmented Generation) knowledge base.
#   Uses separate ChromaDB collections for different data sources with
#   incremental synchronization to avoid redundant processing.
# 
#   Collections:
#     - lisbon_pdf: Static PDF guide (indexed once, never updated)
#     - lisbon_places: VisitLisboa places (weekly sync)
#     - lisbon_events: VisitLisboa events (daily sync)
# 
#   Features:
#     - Incremental sync: only process changed documents
#     - Content hashing to detect modifications
#     - Automatic cleanup of deleted items
#     - Separate collections for independent management
# 
#   Usage:
#     # Full sync (checks all collections, only processes changes)
#     python tools/vector_store.py
#     
#     # Force rebuild specific collection
#     python tools/vector_store.py --rebuild-pdf
#     python tools/vector_store.py --rebuild-places
#     python tools/vector_store.py --rebuild-events
#     
#     # Rebuild everything
#     python tools/vector_store.py --rebuild-all
#     
#     # Test search
#     python tools/vector_store.py --test
# ==========================================================================

# Required libraries:
# pip install langchain-chroma langchain-huggingface pypdf langchain-community tqdm chromadb

import os
import sys

# CRITICAL: Set environment variables BEFORE any imports that may trigger chromadb
# This prevents OpenTelemetry import errors due to version conflicts
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import json
import hashlib
import warnings
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm

# Suppress opentelemetry warnings that may arise from version conflicts
warnings.filterwarnings("ignore", category=DeprecationWarning, module="opentelemetry")
warnings.filterwarnings("ignore", category=ImportWarning)

# Import ChromaDB with error handling for version conflicts
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Add parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config


# ==========================================================================
# Collection Names (Constants)
# ==========================================================================
COLLECTION_PDF = "lisbon_pdf"
COLLECTION_PLACES = "lisbon_places"
COLLECTION_EVENTS = "lisbon_events"


def compute_content_hash(content: str) -> str:
    """
    Computes a SHA-256 hash of the content for change detection.
    
    Args:
        content (str): The text content to hash.
    
    Returns:
        str: The first 16 characters of the SHA-256 hash.
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def generate_doc_id(url: str, source: str) -> str:
    """
    Generates a stable document ID from URL and source.
    
    Args:
        url (str): The document URL (unique identifier).
        source (str): The source tag (places/events).
    
    Returns:
        str: A stable document ID.
    """
    # Use URL hash for stable ID
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
    return f"{source}_{url_hash}"


class KnowledgeBase:
    """
    Manages the RAG knowledge base with incremental synchronization.
    
    This class handles:
        1. Loading and processing static data sources
        2. Creating embeddings using HuggingFace models
        3. Storing vectors in separate ChromaDB collections
        4. Incremental sync: add new, update modified, delete removed
    
    Collections:
        - lisbon_pdf: Static content (indexed once)
        - lisbon_places: Weekly updated places
        - lisbon_events: Daily updated events
    
    Example:
        >>> kb = KnowledgeBase()
        >>> kb.sync_all()  # Incremental sync
        >>> results = kb.search("museums in Lisbon")
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initializes the KnowledgeBase with embedding model.
        
        Args:
            use_gpu (bool): Whether to use GPU for embeddings (requires CUDA).
                           Falls back to CPU if GPU is unavailable.
        """
        print(f"\033[1m📥 Initializing Embeddings:\033[0m {Config.EMBEDDING_MODEL_NAME}...")
        
        # Configure device (GPU if available and requested)
        device = 'cuda' if use_gpu else 'cpu'
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"   \033[1;32m✓ Running on {device.upper()}\033[0m")
        except Exception:
            print(f"   \033[1;33m⚠ GPU unavailable, falling back to CPU\033[0m")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # Path for persistent ChromaDB storage
        self.vector_db_path = str(Config.VECTOR_DB_DIR)
        os.makedirs(self.vector_db_path, exist_ok=True)
    
    # =========================================================================
    # Collection Management
    # =========================================================================
    
    def _get_collection(self, collection_name: str) -> Chroma:
        """
        Gets or creates a ChromaDB collection.
        
        Args:
            collection_name (str): Name of the collection.
        
        Returns:
            Chroma: The ChromaDB collection instance.
        """
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )
    
    def _get_existing_docs(self, collection_name: str) -> Dict[str, str]:
        """
        Gets all existing document IDs and their content hashes from a collection.
        
        Args:
            collection_name (str): Name of the collection.
        
        Returns:
            Dict[str, str]: Mapping of doc_id -> content_hash.
        """
        try:
            vectorstore = self._get_collection(collection_name)
            collection = vectorstore._collection
            
            # Get all documents with metadata
            result = collection.get(include=["metadatas"])
            
            if not result or not result.get("ids"):
                return {}
            
            doc_hashes = {}
            for doc_id, metadata in zip(result["ids"], result["metadatas"]):
                content_hash = metadata.get("content_hash", "")
                doc_hashes[doc_id] = content_hash
            
            return doc_hashes
            
        except Exception:
            # Collection might not exist yet
            return {}
    
    def _delete_collection(self, collection_name: str) -> None:
        """
        Deletes a specific collection from ChromaDB.
        
        Args:
            collection_name (str): Name of the collection to delete.
        """
        try:
            vectorstore = self._get_collection(collection_name)
            vectorstore.delete_collection()
            print(f"   \033[1;33m🗑️ Deleted collection: {collection_name}\033[0m")
        except Exception:
            pass  # Collection might not exist
    
    # =========================================================================
    # Document Processing
    # =========================================================================
    
    def _extract_title(self, item: Dict[str, Any], source_tag: str) -> str:
        """
        Extracts a meaningful title from an item based on source type.
        
        Args:
            item (Dict): The JSON item.
            source_tag (str): Source identifier (places/events).
        
        Returns:
            str: Extracted title.
        """
        # Places have 'title' field
        if 'title' in item and item['title']:
            return item['title']
        
        # Events: extract from URL (e.g., /events/cnb-romeu-e-julieta -> CNB Romeu e Julieta)
        if 'url' in item and item['url']:
            url = item['url']
            # Extract last part of URL path
            slug = url.rstrip('/').split('/')[-1]
            # Clean up: remove hashes, convert dashes to spaces, title case
            slug = slug.split('-')[0:8]  # Limit to first 8 parts
            title = ' '.join(slug).replace('_', ' ').title()
            if title and len(title) > 3:
                return title
        
        # Fallback: use venue_name for events
        if 'venue_name' in item and item['venue_name']:
            return f"Event at {item['venue_name']}"
        
        # Last resort: use first 50 chars of description
        if 'full_description' in item and item['full_description']:
            desc = item['full_description'][:50].strip()
            return f"{desc}..." if len(item['full_description']) > 50 else desc
        
        return 'Unknown'
    
    def _json_to_document(self, item: Dict[str, Any], source_tag: str) -> Tuple[str, Document]:
        """
        Converts a JSON item to a LangChain Document with stable ID.
        
        Args:
            item (Dict): The JSON item from places/events.
            source_tag (str): Source identifier (places/events).
        
        Returns:
            Tuple[str, Document]: (doc_id, Document) pair.
        """
        # Extract title based on source type
        title = self._extract_title(item, source_tag)
        
        # Build rich text representation
        content_parts = [f"Name: {title}"]
        
        for key, value in item.items():
            if key == 'title' or not value:
                continue
            
            if isinstance(value, list):
                val_str = ", ".join(map(str, value))
            elif isinstance(value, dict):
                val_str = json.dumps(value, ensure_ascii=False)
            else:
                val_str = str(value)
            
            key_clean = key.replace('_', ' ').title()
            content_parts.append(f"{key_clean}: {val_str}")
        
        content = "\n".join(content_parts)
        content_hash = compute_content_hash(content)
        
        # Generate stable ID from URL
        url = item.get('url', item.get('title', str(hash(content))))
        doc_id = generate_doc_id(url, source_tag)
        
        metadata = {
            "source": source_tag,
            "title": title,
            "url": item.get('url', ''),
            "category": item.get('category', 'General'),
            "content_hash": content_hash,
            "indexed_at": datetime.now().isoformat()
        }
        
        doc = Document(page_content=content, metadata=metadata)
        return doc_id, doc
    
    def _load_json_data(self, file_path: str, source_tag: str) -> Dict[str, Document]:
        """
        Loads JSON data and converts to Documents with stable IDs.
        
        Args:
            file_path (str): Path to the JSON file.
            source_tag (str): Source identifier.
        
        Returns:
            Dict[str, Document]: Mapping of doc_id -> Document.
        """
        if not os.path.exists(file_path):
            print(f"\033[1;33m⚠️ Warning:\033[0m File not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        docs = {}
        for item in data:
            doc_id, doc = self._json_to_document(item, source_tag)
            docs[doc_id] = doc
        
        return docs
    
    # =========================================================================
    # PDF Collection (Static, One-time Index)
    # =========================================================================
    
    def sync_pdf_collection(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Syncs the PDF collection. Only indexes if empty or forced.
        
        Args:
            force_rebuild (bool): If True, deletes and rebuilds the collection.
        
        Returns:
            Dict[str, int]: Statistics about the sync operation.
        """
        print(f"\n\033[1m📚 PDF Collection ({COLLECTION_PDF})\033[0m")
        
        if force_rebuild:
            self._delete_collection(COLLECTION_PDF)
        
        # Check if collection already has data
        existing = self._get_existing_docs(COLLECTION_PDF)
        
        if existing and not force_rebuild:
            print(f"   \033[1;32m✓ Already indexed ({len(existing)} chunks). Skipping.\033[0m")
            return {"status": "skipped", "existing": len(existing)}
        
        # Load and process PDF
        pdf_path = str(Config.PATH_PDF_TEXT)
        if not os.path.exists(pdf_path):
            print(f"   \033[1;33m⚠️ PDF not found: {pdf_path}\033[0m")
            return {"status": "error", "error": "PDF not found"}
        
        print(f"   📖 Loading: {os.path.basename(pdf_path)}")
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        docs = text_splitter.split_documents(pages)
        
        # Extract PDF base name for title (e.g., "TurismodeLisboa_OfficialGuide")
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_title = pdf_basename.replace('_', ' ').replace('-', ' ').title()
        
        # Add metadata and generate IDs
        doc_ids = []
        for i, doc in enumerate(docs):
            doc_id = f"pdf_chunk_{i:04d}"
            doc_ids.append(doc_id)
            
            # Extract page number from PyPDF metadata
            page_num = doc.metadata.get('page', i)
            
            doc.metadata["source"] = "TurismoLisboa_OfficialGuide_PDF"
            doc.metadata["title"] = f"{pdf_title} (p.{page_num + 1})"
            doc.metadata["url"] = f"{os.path.basename(pdf_path)}#page={page_num + 1}"
            doc.metadata["category"] = "Official Guide"
            doc.metadata["page"] = page_num + 1  # Store 1-indexed page number
            doc.metadata["content_hash"] = compute_content_hash(doc.page_content)
            doc.metadata["indexed_at"] = datetime.now().isoformat()
        
        print(f"   📊 Indexing {len(docs)} chunks...")
        
        # Create collection with all documents
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=COLLECTION_PDF,
            persist_directory=self.vector_db_path,
            ids=doc_ids
        )
        
        print(f"   \033[1;32m✓ Indexed {len(docs)} PDF chunks\033[0m")
        return {"status": "indexed", "added": len(docs)}
    
    # =========================================================================
    # JSON Collection Sync (Incremental)
    # =========================================================================
    
    def _sync_json_collection(
        self, 
        collection_name: str, 
        json_path: str, 
        source_tag: str,
        force_rebuild: bool = False
    ) -> Dict[str, int]:
        """
        Performs incremental sync for a JSON-based collection.
        
        This method:
            1. Loads current JSON data
            2. Compares with existing collection (by content hash)
            3. Adds new documents
            4. Updates modified documents (delete + add)
            5. Removes deleted documents
        
        Args:
            collection_name (str): Name of the ChromaDB collection.
            json_path (str): Path to the JSON file.
            source_tag (str): Source identifier for metadata.
            force_rebuild (bool): If True, rebuilds from scratch.
        
        Returns:
            Dict[str, int]: Statistics about the sync operation.
        """
        print(f"\n\033[1m📁 {source_tag} Collection ({collection_name})\033[0m")
        
        if force_rebuild:
            self._delete_collection(collection_name)
        
        # Load current JSON data
        current_docs = self._load_json_data(json_path, source_tag)
        
        if not current_docs:
            print(f"   \033[1;33m⚠️ No data loaded from {json_path}\033[0m")
            return {"status": "error", "error": "No data loaded"}
        
        print(f"   📂 Loaded {len(current_docs)} items from JSON")
        
        # Get existing documents from collection
        existing_hashes = self._get_existing_docs(collection_name)
        print(f"   📊 Existing in DB: {len(existing_hashes)} items")
        
        # Compute diff
        current_ids = set(current_docs.keys())
        existing_ids = set(existing_hashes.keys())
        
        # New documents (in JSON but not in DB)
        new_ids = current_ids - existing_ids
        
        # Deleted documents (in DB but not in JSON)
        deleted_ids = existing_ids - current_ids
        
        # Modified documents (in both, but hash changed)
        modified_ids = set()
        for doc_id in current_ids & existing_ids:
            current_hash = current_docs[doc_id].metadata.get("content_hash", "")
            existing_hash = existing_hashes.get(doc_id, "")
            if current_hash != existing_hash:
                modified_ids.add(doc_id)
        
        # Report diff
        print(f"   \033[1;32m➕ New:\033[0m {len(new_ids)}")
        print(f"   \033[1;33m🔄 Modified:\033[0m {len(modified_ids)}")
        print(f"   \033[1;31m➖ Deleted:\033[0m {len(deleted_ids)}")
        
        # Skip if no changes
        if not new_ids and not modified_ids and not deleted_ids:
            print(f"   \033[1;32m✓ No changes detected. Collection is up-to-date.\033[0m")
            return {
                "status": "no_changes",
                "existing": len(existing_ids)
            }
        
        # Get or create collection
        vectorstore = self._get_collection(collection_name)
        collection = vectorstore._collection
        
        # Delete removed and modified documents
        ids_to_delete = list(deleted_ids | modified_ids)
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"   🗑️ Deleted {len(ids_to_delete)} documents from DB")
        
        # Add new and modified documents
        ids_to_add = list(new_ids | modified_ids)
        if ids_to_add:
            docs_to_add = [current_docs[doc_id] for doc_id in ids_to_add]
            
            # Process in batches with progress bar
            batch_size = 50
            total_batches = (len(docs_to_add) + batch_size - 1) // batch_size
            
            if total_batches > 1:
                pbar = tqdm(
                    total=len(docs_to_add),
                    desc="   🔄 Indexing",
                    unit="docs",
                )
            
            for i in range(0, len(docs_to_add), batch_size):
                batch_docs = docs_to_add[i:i + batch_size]
                batch_ids = ids_to_add[i:i + batch_size]
                
                vectorstore.add_documents(batch_docs, ids=batch_ids)
                
                if total_batches > 1:
                    pbar.update(len(batch_docs))
            
            if total_batches > 1:
                pbar.close()
            
            print(f"   \033[1;32m✓ Added/Updated {len(ids_to_add)} documents\033[0m")
        
        return {
            "status": "synced",
            "added": len(new_ids),
            "modified": len(modified_ids),
            "deleted": len(deleted_ids),
            "total": len(current_ids)
        }
    
    def sync_places_collection(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Syncs the places collection with incremental updates.
        
        Args:
            force_rebuild (bool): If True, rebuilds from scratch.
        
        Returns:
            Dict[str, int]: Sync statistics.
        """
        return self._sync_json_collection(
            collection_name=COLLECTION_PLACES,
            json_path=str(Config.PATH_VISIT_LISBOA_PLACES),
            source_tag="VisitLisboa_Places",
            force_rebuild=force_rebuild
        )
    
    def sync_events_collection(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Syncs the events collection with incremental updates.
        
        Args:
            force_rebuild (bool): If True, rebuilds from scratch.
        
        Returns:
            Dict[str, int]: Sync statistics.
        """
        return self._sync_json_collection(
            collection_name=COLLECTION_EVENTS,
            json_path=str(Config.PATH_VISIT_LISBOA_EVENTS),
            source_tag="VisitLisboa_Events",
            force_rebuild=force_rebuild
        )
    
    # =========================================================================
    # Full Sync
    # =========================================================================
    
    def sync_all(
        self, 
        rebuild_pdf: bool = False,
        rebuild_places: bool = False,
        rebuild_events: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronizes all collections with incremental updates.
        
        Args:
            rebuild_pdf (bool): Force rebuild PDF collection.
            rebuild_places (bool): Force rebuild places collection.
            rebuild_events (bool): Force rebuild events collection.
        
        Returns:
            Dict[str, Any]: Statistics for all collections.
        """
        print("\033[1m" + "=" * 60 + "\033[0m")
        print("\033[1m🔄 Vector Store Incremental Sync\033[0m")
        print("\033[1m" + "=" * 60 + "\033[0m")
        
        results = {}
        
        # Sync PDF (static, skip if exists)
        results["pdf"] = self.sync_pdf_collection(force_rebuild=rebuild_pdf)
        
        # Sync Places (weekly updates)
        results["places"] = self.sync_places_collection(force_rebuild=rebuild_places)
        
        # Sync Events (daily updates)
        results["events"] = self.sync_events_collection(force_rebuild=rebuild_events)
        
        # Summary
        print("\n\033[1m" + "=" * 60 + "\033[0m")
        print("\033[1m📊 Sync Summary\033[0m")
        print("\033[1m" + "=" * 60 + "\033[0m")
        
        for name, stats in results.items():
            status = stats.get("status", "unknown")
            if status == "no_changes":
                print(f"   {name}: \033[1;32m✓ No changes\033[0m ({stats.get('existing', 0)} docs)")
            elif status == "skipped":
                print(f"   {name}: \033[1;32m✓ Skipped\033[0m ({stats.get('existing', 0)} docs)")
            elif status == "synced":
                added = stats.get("added", 0)
                modified = stats.get("modified", 0)
                deleted = stats.get("deleted", 0)
                total = stats.get("total", 0)
                print(f"   {name}: \033[1;32m✓ Synced\033[0m (+{added} ~{modified} -{deleted} = {total} docs)")
            elif status == "indexed":
                print(f"   {name}: \033[1;32m✓ Indexed\033[0m ({stats.get('added', 0)} docs)")
            else:
                print(f"   {name}: \033[1;31m✗ {status}\033[0m")
        
        return results
    
    # =========================================================================
    # Search and Retrieval
    # =========================================================================
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        collections: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Performs semantic search across collections.
        
        Args:
            query (str): Search query in natural language.
            k (int): Number of results to return per collection.
            collections (List[str]): Collections to search. If None, searches all.
        
        Returns:
            List[Document]: Most relevant documents, sorted by relevance.
        """
        if collections is None:
            collections = [COLLECTION_PDF, COLLECTION_PLACES, COLLECTION_EVENTS]
        
        all_results = []
        
        for col_name in collections:
            try:
                vectorstore = self._get_collection(col_name)
                results = vectorstore.similarity_search_with_score(query, k=k)
                all_results.extend(results)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Search error in {col_name}: {e}")
                continue
        
        # Sort by score (lower is better for distance)
        all_results.sort(key=lambda x: x[1])
        
        # Return top k documents
        return [doc for doc, score in all_results[:k]]
    
    def get_retriever(self, k: int = 5):
        """
        Returns a retriever that searches across all collections.
        
        Args:
            k (int): Number of documents to retrieve.
        
        Returns:
            VectorStoreRetriever: Combined retriever.
        
        Notes:
            This returns a retriever for the events collection by default,
            as it's most frequently updated and relevant for current queries.
            For comprehensive search, use the search() method directly.
        """
        # Return retriever for the most dynamic collection
        vectorstore = self._get_collection(COLLECTION_EVENTS)
        return vectorstore.as_retriever(search_kwargs={"k": k})
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about all collections.
        
        Returns:
            Dict[str, Any]: Statistics for each collection.
        """
        stats = {}
        
        for col_name in [COLLECTION_PDF, COLLECTION_PLACES, COLLECTION_EVENTS]:
            try:
                vectorstore = self._get_collection(col_name)
                count = vectorstore._collection.count()
                stats[col_name] = {"status": "ready", "count": count}
            except Exception:
                stats[col_name] = {"status": "not_built", "count": 0}
        
        stats["total"] = sum(s.get("count", 0) for s in stats.values() if isinstance(s, dict))
        stats["path"] = self.vector_db_path
        
        return stats


# ==========================================================================
# CLI Interface
# ==========================================================================
if __name__ == "__main__":
    """
    Command-line interface for vector store management.
    
    Usage:
        python tools/vector_store.py                    # Incremental sync
        python tools/vector_store.py --rebuild-all      # Rebuild everything
        python tools/vector_store.py --rebuild-pdf      # Rebuild PDF only
        python tools/vector_store.py --rebuild-places   # Rebuild places only
        python tools/vector_store.py --rebuild-events   # Rebuild events only
        python tools/vector_store.py --test             # Test search
        python tools/vector_store.py --stats            # Show statistics
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Vector Store Management with Incremental Sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/vector_store.py                    # Incremental sync (default)
  python tools/vector_store.py --rebuild-all      # Force rebuild everything
  python tools/vector_store.py --rebuild-events   # Rebuild events only
  python tools/vector_store.py --test             # Test search functionality
  python tools/vector_store.py --stats            # Show database statistics
        """
    )
    
    parser.add_argument(
        "--rebuild-all", 
        action="store_true", 
        help="Force rebuild all collections"
    )
    parser.add_argument(
        "--rebuild-pdf", 
        action="store_true", 
        help="Force rebuild PDF collection"
    )
    parser.add_argument(
        "--rebuild-places", 
        action="store_true", 
        help="Force rebuild places collection"
    )
    parser.add_argument(
        "--rebuild-events", 
        action="store_true", 
        help="Force rebuild events collection"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test search functionality"
    )
    parser.add_argument(
        "--stats", 
        action="store_true", 
        help="Show database statistics"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU (use CPU only)"
    )
    
    args = parser.parse_args()
    
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Vector Store Management (Incremental Sync)\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    # Initialize knowledge base
    kb = KnowledgeBase(use_gpu=not args.no_gpu)
    
    if args.stats:
        # Show statistics only
        print("\n\033[1m📊 Database Statistics\033[0m")
        stats = kb.get_stats()
        print(f"   Path: {stats['path']}")
        print(f"   Total documents: {stats['total']}")
        for col_name in [COLLECTION_PDF, COLLECTION_PLACES, COLLECTION_EVENTS]:
            col_stats = stats.get(col_name, {})
            status = col_stats.get("status", "unknown")
            count = col_stats.get("count", 0)
            print(f"   - {col_name}: {count} docs ({status})")
    
    elif args.test:
        # Comprehensive metadata validation test
        print("\n\033[1m🔍 Testing Vector Store...\033[0m")
        stats = kb.get_stats()
        print(f"   Total documents: {stats['total']}")
        
        if stats['total'] == 0:
            print("   \033[1;33m⚠️ Database is empty. Run sync first.\033[0m")
        else:
            # Required metadata fields per source
            required_fields = {
                "TurismoLisboa_OfficialGuide_PDF": ["source", "title", "url", "category", "content_hash", "indexed_at", "page"],
                "VisitLisboa_Places": ["source", "title", "url", "category", "content_hash", "indexed_at"],
                "VisitLisboa_Events": ["source", "title", "url", "category", "content_hash", "indexed_at"]
            }
            
            # Test each collection
            print("\n\033[1m📋 Metadata Validation by Collection:\033[0m")
            
            for col_name, source_name in [
                (COLLECTION_PDF, "TurismoLisboa_OfficialGuide_PDF"),
                (COLLECTION_PLACES, "VisitLisboa_Places"),
                (COLLECTION_EVENTS, "VisitLisboa_Events")
            ]:
                try:
                    vectorstore = kb._get_collection(col_name)
                    collection = vectorstore._collection
                    result = collection.get(include=["metadatas", "documents"], limit=5)
                    
                    if not result or not result.get("ids"):
                        print(f"\n   \033[1;33m⚠️ {col_name}: Empty collection\033[0m")
                        continue
                    
                    count = collection.count()
                    print(f"\n   \033[1m{col_name}\033[0m ({count} docs)")
                    
                    # Check first 3 documents
                    sample_size = min(3, len(result["ids"]))
                    missing_fields = set()
                    empty_fields = set()
                    
                    for i in range(sample_size):
                        metadata = result["metadatas"][i]
                        doc_content = result["documents"][i][:100] + "..." if len(result["documents"][i]) > 100 else result["documents"][i]
                        
                        # Check required fields
                        for field in required_fields.get(source_name, []):
                            if field not in metadata:
                                missing_fields.add(field)
                            else:
                                val = metadata[field]
                                # Handle numeric fields (page can be 0) vs string fields
                                if isinstance(val, (int, float)):
                                    continue  # Numeric values are valid (even 0)
                                elif not val or val in ["N/A", "Unknown", ""]:
                                    empty_fields.add(field)
                        
                        # Display sample
                        title = metadata.get('title', 'N/A')[:60]
                        url = metadata.get('url', 'N/A')[:50]
                        category = metadata.get('category', 'N/A')
                        print(f"      {i+1}. Title: \033[1;36m{title}\033[0m")
                        print(f"         URL: {url}")
                        print(f"         Category: {category}")
                    
                    # Report field issues
                    if missing_fields:
                        print(f"      \033[1;31m❌ Missing fields: {', '.join(missing_fields)}\033[0m")
                    if empty_fields:
                        print(f"      \033[1;33m⚠️ Empty/Invalid fields: {', '.join(empty_fields)}\033[0m")
                    if not missing_fields and not empty_fields:
                        print(f"      \033[1;32m✓ All metadata fields valid\033[0m")
                        
                except Exception as e:
                    print(f"\n   \033[1;31m❌ {col_name}: Error - {e}\033[0m")
            
            # Search test
            print("\n\033[1m🔍 Search Quality Test:\033[0m")
            test_queries = [
                ("museums in Belém", "Should return places/PDF about Belém museums"),
                ("traditional Portuguese food", "Should return restaurants"),
                ("events this week", "Should return events"),
                ("metro transport", "Should return transport info from PDF/places")
            ]
            
            for query, expected in test_queries:
                print(f"\n   \033[1m📝 Query:\033[0m {query}")
                print(f"      Expected: {expected}")
                results = kb.search(query, k=3)
                for i, doc in enumerate(results, 1):
                    title = doc.metadata.get('title', 'N/A')[:50]
                    source = doc.metadata.get('source', 'N/A')
                    score_indicator = "✓" if title != "N/A" and title != "Unknown" else "✗"
                    print(f"      {i}. [{score_indicator}] {title} ({source})")
    
    else:
        # Sync mode
        rebuild_all = args.rebuild_all
        
        kb.sync_all(
            rebuild_pdf=rebuild_all or args.rebuild_pdf,
            rebuild_places=rebuild_all or args.rebuild_places,
            rebuild_events=rebuild_all or args.rebuild_events
        )
        
        # Show final stats
        print("\n\033[1m📊 Final Statistics\033[0m")
        stats = kb.get_stats()
        print(f"   Total documents: {stats['total']}")
        for col_name in [COLLECTION_PDF, COLLECTION_PLACES, COLLECTION_EVENTS]:
            col_stats = stats.get(col_name, {})
            count = col_stats.get("count", 0)
            print(f"   - {col_name}: {count} docs")
