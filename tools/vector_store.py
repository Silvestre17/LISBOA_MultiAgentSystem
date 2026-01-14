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

import sys
import os
import signal

# 🚀 CRITICAL: Force unbuffered output immediately to debug GitHub Actions hangs
sys.stdout.reconfigure(line_buffering=True)

# Set environment variables BEFORE any heavy imports
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import json
import hashlib
import warnings
import argparse
import gc
import time
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime
from tqdm import tqdm

# ==========================================================================
# Signal Handling (Exit Code 143 = SIGTERM from GitHub Actions)
# ==========================================================================
_graceful_exit_requested = False

def _sigterm_handler(signum, frame):
    """Handle SIGTERM gracefully to avoid exit code 143."""
    global _graceful_exit_requested
    _graceful_exit_requested = True
    print("\n\033[1;33m⚠️  SIGTERM received - Gracefully exiting...\033[0m", flush=True)
    print("   Will complete current batch and exit with code 2 (more work pending)", flush=True)

# Register handler for SIGTERM (signal 15)
signal.signal(signal.SIGTERM, _sigterm_handler)
# Also handle SIGINT (Ctrl+C) for local testing
signal.signal(signal.SIGINT, _sigterm_handler)

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

# Lazy imports for heavy libraries to prevent startup hangs
if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

# ==========================================================================
# Constants
# ==========================================================================
COLLECTION_PDF = "lisbon_pdf"
COLLECTION_PLACES = "lisbon_places"
COLLECTION_EVENTS = "lisbon_events"


def compute_content_hash(content: str) -> str:
    """
    Computes a SHA-256 hash of the content string.

    Args:
        content (str): The text content to hash.

    Returns:
        str: The first 16 characters of the hex digest.
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def generate_doc_id(url: str, source: str) -> str:
    """
    Generates a stable document ID based on URL and source.

    Args:
        url (str): The URL or unique identifier of the item.
        source (str): The source tag (e.g., 'VisitLisboa_Places').

    Returns:
        str: A unique document ID string.
    """
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
    return f"{source}_{url_hash}"


class KnowledgeBase:
    """
    Manages the RAG knowledge base with incremental synchronization capabilities.
    
    Handles initialization of embeddings, vector store connections, and 
    synchronization logic for different data sources.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initializes the KnowledgeBase.

        Args:
            use_gpu (bool): Whether to attempt using GPU for embeddings. Defaults to True.
        """
        print(f"\033[1m📥 Initializing KnowledgeBase...\033[0m", flush=True)
        
        # Lazy import heavy libraries here
        print("   Importing AI libraries (this may take a moment)...", flush=True)
        global Chroma, HuggingFaceEmbeddings, Document, RecursiveCharacterTextSplitter, PyPDFLoader
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        
        print(f"   Loading Embeddings: {Config.EMBEDDING_MODEL_NAME}...", flush=True)
        
        device = 'cuda' if use_gpu else 'cpu'
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"   \033[1;32m✓ Embeddings ready on {device.upper()}\033[0m", flush=True)
        except Exception as e:
            print(f"   \033[1;33m⚠ GPU error: {e}. Falling back to CPU.\033[0m", flush=True)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.vector_db_path = str(Config.VECTOR_DB_DIR)
        os.makedirs(self.vector_db_path, exist_ok=True)
        print(f"   DB Path: {self.vector_db_path}", flush=True)

    def _get_collection(self, collection_name: str) -> 'Chroma':
        """
        Retrieves a ChromaDB collection object.

        Args:
            collection_name (str): The name of the collection to retrieve.

        Returns:
            Chroma: The ChromaDB collection object.
        """
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )
    
    def _get_existing_docs(self, collection_name: str) -> Dict[str, str]:
        """
        Retrieves existing document IDs and their content hashes from a collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            Dict[str, str]: A dictionary mapping document IDs to their content hashes.
        """
        try:
            vectorstore = self._get_collection(collection_name)
            # Access the underlying chromadb collection directly for speed
            collection = vectorstore._collection
            result = collection.get(include=["metadatas"])
            
            if not result or not result.get("ids"):
                return {}
            
            doc_hashes = {}
            for doc_id, metadata in zip(result["ids"], result["metadatas"]):
                if metadata:
                    doc_hashes[doc_id] = metadata.get("content_hash", "")
            return doc_hashes
        except Exception:
            return {}

    def _delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection from the vector database.

        Args:
            collection_name (str): The name of the collection to delete.
        """
        try:
            vectorstore = self._get_collection(collection_name)
            vectorstore.delete_collection()
            print(f"   \033[1;33m🗑️ Deleted collection: {collection_name}\033[0m", flush=True)
        except Exception:
            pass

    def _extract_title(self, item: Dict[str, Any], source_tag: str) -> str:
        """
        Extracts a meaningful title from a data item.

        Args:
            item (Dict[str, Any]): The data item.
            source_tag (str): The source tag.

        Returns:
            str: The extracted title or 'Unknown'.
        """
        if 'title' in item and item['title']:
            return item['title']
        
        if 'url' in item and item['url']:
            slug = item['url'].rstrip('/').split('/')[-1]
            slug = slug.split('-')[0:8]
            title = ' '.join(slug).replace('_', ' ').title()
            if title and len(title) > 3:
                return title
        
        if 'venue_name' in item and item['venue_name']:
            return f"Event at {item['venue_name']}"
        
        if 'full_description' in item and item['full_description']:
            desc = item['full_description'][:50].strip()
            return f"{desc}..." if len(item['full_description']) > 50 else desc
        
        return 'Unknown'

    def _json_to_document(self, item: Dict[str, Any], source_tag: str) -> Tuple[str, 'Document']:
        """
        Converts a JSON item into a LangChain Document.

        Args:
            item (Dict[str, Any]): The JSON item.
            source_tag (str): The source tag.

        Returns:
            Tuple[str, Document]: A tuple containing the document ID and the Document object.
        """
        title = self._extract_title(item, source_tag)
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
            content_parts.append(f"{key.replace('_', ' ').title()}: {val_str}")
        
        content = "\n".join(content_parts)
        content_hash = compute_content_hash(content)
        
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
        return doc_id, Document(page_content=content, metadata=metadata)

    def _load_json_data(self, file_path: str, source_tag: str) -> Dict[str, 'Document']:
        """
        Loads JSON data from a file and converts it to Documents.

        Args:
            file_path (str): Path to the JSON file.
            source_tag (str): The source tag.

        Returns:
            Dict[str, Document]: A dictionary mapping document IDs to Document objects.
        """
        if not os.path.exists(file_path):
            print(f"\033[1;33m⚠️ Warning:\033[0m File not found: {file_path}", flush=True)
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"\033[1;31m❌ Error reading JSON {file_path}: {e}\033[0m", flush=True)
            return {}
        
        docs = {}
        for item in data:
            doc_id, doc = self._json_to_document(item, source_tag)
            docs[doc_id] = doc
        return docs

    def sync_pdf_collection(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Synchronizes the PDF collection.

        Args:
            force_rebuild (bool): Whether to force a full rebuild of the collection.

        Returns:
            Dict[str, int]: Statistics about the sync operation.
        """
        print(f"\n\033[1m📚 PDF Collection ({COLLECTION_PDF})\033[0m", flush=True)
        
        if force_rebuild:
            self._delete_collection(COLLECTION_PDF)
        
        existing = self._get_existing_docs(COLLECTION_PDF)
        if existing and not force_rebuild:
            print(f"   \033[1;32m✓ Already indexed ({len(existing)} chunks). Skipping.\033[0m", flush=True)
            return {"status": "skipped", "existing": len(existing)}
        
        pdf_path = str(Config.PATH_PDF_TEXT)
        if not os.path.exists(pdf_path):
            print(f"   \033[1;33m⚠️ PDF not found: {pdf_path}\033[0m", flush=True)
            return {"status": "error", "error": "PDF not found"}
        
        print(f"   📖 Loading: {os.path.basename(pdf_path)}", flush=True)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        docs = text_splitter.split_documents(pages)
        
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_title = pdf_basename.replace('_', ' ').replace('-', ' ').title()
        
        doc_ids = []
        for i, doc in enumerate(docs):
            doc_id = f"pdf_chunk_{i:04d}"
            doc_ids.append(doc_id)
            page_num = doc.metadata.get('page', i)
            doc.metadata["source"] = "TurismoLisboa_OfficialGuide_PDF"
            doc.metadata["title"] = f"{pdf_title} (p.{page_num + 1})"
            doc.metadata["url"] = f"{os.path.basename(pdf_path)}#page={page_num + 1}"
            doc.metadata["category"] = "Official Guide"
            doc.metadata["page"] = page_num + 1
            doc.metadata["content_hash"] = compute_content_hash(doc.page_content)
            doc.metadata["indexed_at"] = datetime.now().isoformat()
        
        print(f"   📊 Indexing {len(docs)} chunks...", flush=True)
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=COLLECTION_PDF,
            persist_directory=self.vector_db_path,
            ids=doc_ids
        )
        
        print(f"   \033[1;32m✓ Indexed {len(docs)} PDF chunks\033[0m", flush=True)
        return {"status": "indexed", "added": len(docs)}

    def _sync_json_collection(
        self, 
        collection_name: str, 
        json_path: str, 
        source_tag: str,
        force_rebuild: bool = False,
        max_docs: int = None
    ) -> Dict[str, int]:
        """
        Synchronizes a JSON-based collection (Places or Events).

        Args:
            collection_name (str): The name of the collection.
            json_path (str): Path to the JSON source file.
            source_tag (str): The source tag for documents.
            force_rebuild (bool): Whether to force a full rebuild.
            max_docs (int, optional): Maximum number of documents to process.

        Returns:
            Dict[str, int]: Statistics about the sync operation.
        """
        print(f"\n\033[1m📁 {source_tag} Collection ({collection_name})\033[0m", flush=True)
        
        if force_rebuild:
            self._delete_collection(collection_name)
        
        current_docs = self._load_json_data(json_path, source_tag)
        if not current_docs:
            print(f"   \033[1;33m⚠️ No data loaded from {json_path}\033[0m", flush=True)
            return {"status": "error", "error": "No data loaded"}
        
        print(f"   📂 Loaded {len(current_docs)} items from JSON", flush=True)
        
        existing_hashes = self._get_existing_docs(collection_name)
        print(f"   📊 Existing in DB: {len(existing_hashes)} items", flush=True)
        
        current_ids = set(current_docs.keys())
        existing_ids = set(existing_hashes.keys())
        
        new_ids = current_ids - existing_ids
        deleted_ids = existing_ids - current_ids
        modified_ids = {
            doc_id for doc_id in current_ids & existing_ids
            if current_docs[doc_id].metadata.get("content_hash", "") != existing_hashes.get(doc_id, "")
        }
        
        print(f"   \033[1;32m➕ New:\033[0m {len(new_ids)}", flush=True)
        print(f"   \033[1;33m🔄 Modified:\033[0m {len(modified_ids)}", flush=True)
        print(f"   \033[1;31m➖ Deleted:\033[0m {len(deleted_ids)}", flush=True)
        
        if not new_ids and not modified_ids and not deleted_ids:
            print(f"   \033[1;32m✓ No changes detected.\033[0m", flush=True)
            return {"status": "no_changes", "existing": len(existing_ids)}
        
        vectorstore = self._get_collection(collection_name)
        collection = vectorstore._collection
        
        ids_to_delete = list(deleted_ids | modified_ids)
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"   🗑️ Deleted {len(ids_to_delete)} documents from DB", flush=True)
        
        ids_to_add = list(new_ids | modified_ids)
        has_more_work = False
        
        if max_docs and len(ids_to_add) > max_docs:
            print(f"   ⚠️ Limiting to {max_docs} documents (out of {len(ids_to_add)})", flush=True)
            ids_to_add = ids_to_add[:max_docs]
            has_more_work = True
        
        if ids_to_add:
            docs_to_add = [current_docs[doc_id] for doc_id in ids_to_add]
            # Reduced from 20 to 10 to prevent OOM/SIGTERM on GitHub Actions
            batch_size = 10
            
            print(f"   🔄 Indexing {len(docs_to_add)} documents (batch size: {batch_size})...", flush=True)
            
            # Use tqdm always to show progress in logs
            iterator = range(0, len(docs_to_add), batch_size)
            iterator = tqdm(
                iterator, 
                total=(len(docs_to_add) + batch_size - 1) // batch_size, 
                desc="   Batch",
                file=sys.stdout,
                mininterval=1.0
            )
            
            processed_count = 0
            for i in iterator:
                # Check for graceful exit signal (SIGTERM from GitHub Actions)
                if _graceful_exit_requested:
                    print(f"\n   \033[1;33m⚠️ Graceful exit: Processed {processed_count}/{len(docs_to_add)} docs\033[0m", flush=True)
                    has_more_work = True
                    break
                
                batch_docs = docs_to_add[i:i + batch_size]
                batch_ids = ids_to_add[i:i + batch_size]
                vectorstore.add_documents(batch_docs, ids=batch_ids)
                processed_count += len(batch_docs)
                
                # 🧹 Force garbage collection to free memory
                gc.collect()
                # ⏳ Sleep briefly to let CPU cool down and reduce resource pressure
                time.sleep(0.5)
            
            if not _graceful_exit_requested:
                print(f"   \033[1;32m✓ Added/Updated {len(ids_to_add)} documents\033[0m", flush=True)
        
        return {
            "status": "synced",
            "added": len([x for x in ids_to_add if x in new_ids]),
            "modified": len([x for x in ids_to_add if x in modified_ids]),
            "deleted": len(deleted_ids),
            "total": len(current_ids),
            "has_more_work": has_more_work,
            "pending": len(new_ids | modified_ids) - len(ids_to_add) if has_more_work else 0
        }

    def sync_places_collection(self, force_rebuild: bool = False, max_docs: int = None) -> Dict[str, int]:
        """Synchronizes the VisitLisboa Places collection."""
        return self._sync_json_collection(
            COLLECTION_PLACES, str(Config.PATH_VISIT_LISBOA_PLACES), "VisitLisboa_Places", force_rebuild, max_docs
        )
    
    def sync_events_collection(self, force_rebuild: bool = False, max_docs: int = None) -> Dict[str, int]:
        """Synchronizes the VisitLisboa Events collection."""
        return self._sync_json_collection(
            COLLECTION_EVENTS, str(Config.PATH_VISIT_LISBOA_EVENTS), "VisitLisboa_Events", force_rebuild, max_docs
        )

    def sync_all(self, rebuild_pdf: bool = False, rebuild_places: bool = False, 
                 rebuild_events: bool = False, max_docs: int = None) -> Dict[str, Any]:
        """
        Runs synchronization for all collections.

        Args:
            rebuild_pdf (bool): Force rebuild PDF collection.
            rebuild_places (bool): Force rebuild Places collection.
            rebuild_events (bool): Force rebuild Events collection.
            max_docs (int, optional): Max docs to process per collection.

        Returns:
            Dict[str, Any]: Summary of synchronization results.
        """
        print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
        print("\033[1m🔄 Vector Store Incremental Sync\033[0m", flush=True)
        if max_docs:
            print(f"\033[1m   (Max {max_docs} docs per collection)\033[0m", flush=True)
        print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
        
        results = {}
        has_more_work = False
        
        results["pdf"] = self.sync_pdf_collection(force_rebuild=rebuild_pdf)
        
        # Check for graceful exit signal between collections
        if _graceful_exit_requested:
            print("\n\033[1;33m⚠️ Graceful exit requested, skipping remaining collections\033[0m", flush=True)
            results["has_more_work"] = True
            return results
        
        results["places"] = self.sync_places_collection(force_rebuild=rebuild_places, max_docs=max_docs)
        if results["places"].get("has_more_work"):
            has_more_work = True
        
        # Check for graceful exit signal between collections
        if _graceful_exit_requested:
            print("\n\033[1;33m⚠️ Graceful exit requested, skipping remaining collections\033[0m", flush=True)
            has_more_work = True
        else:
            events_max_docs = max_docs
            results["events"] = self.sync_events_collection(force_rebuild=rebuild_events, max_docs=events_max_docs)
            if results["events"].get("has_more_work"):
                has_more_work = True
            
        print("\n\033[1m" + "=" * 60 + "\033[0m", flush=True)
        print("\033[1m📊 Sync Summary\033[0m", flush=True)
        print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
        
        for name, stats in results.items():
            status = stats.get("status", "unknown")
            if status == "no_changes":
                print(f"   {name}: \033[1;32m✓ No changes\033[0m ({stats.get('existing', 0)} docs)", flush=True)
            elif status == "synced":
                added = stats.get("added", 0)
                modified = stats.get("modified", 0)
                deleted = stats.get("deleted", 0)
                pending = stats.get("pending", 0)
                msg = f"   {name}: \033[1;32m✓ Synced\033[0m (+{added} ~{modified} -{deleted})"
                if pending > 0:
                    msg += f" \033[1;33m({pending} pending)\033[0m"
                print(msg, flush=True)
            elif status == "indexed":
                print(f"   {name}: \033[1;32m✓ Indexed\033[0m ({stats.get('added', 0)} docs)", flush=True)
            elif status == "skipped":
                print(f"   {name}: \033[1;32m✓ Skipped\033[0m", flush=True)
            else:
                print(f"   {name}: \033[1;31m✗ {status}\033[0m", flush=True)
        
        results["has_more_work"] = has_more_work
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves statistics about the vector store collections.

        Returns:
            Dict[str, Any]: A dictionary containing counts and status for each collection.
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

    def search(self, query: str, k: int = 5, collections: Optional[List[str]] = None) -> List['Document']:
        """
        Searches the knowledge base for relevant documents.

        Args:
            query (str): The search query.
            k (int): Number of results to return.
            collections (List[str], optional): List of collections to search.

        Returns:
            List[Document]: A list of matching documents sorted by relevance.
        """
        if collections is None:
            collections = [COLLECTION_PDF, COLLECTION_PLACES, COLLECTION_EVENTS]
        all_results = []
        for col_name in collections:
            try:
                vectorstore = self._get_collection(col_name)
                results = vectorstore.similarity_search_with_score(query, k=k)
                all_results.extend(results)
            except Exception:
                continue
        all_results.sort(key=lambda x: x[1])
        return [doc for doc, score in all_results[:k]]


if __name__ == "__main__":
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
    
    parser.add_argument("--rebuild-all", action="store_true", help="Force rebuild all collections")
    parser.add_argument("--rebuild-pdf", action="store_true", help="Force rebuild PDF collection")
    parser.add_argument("--rebuild-places", action="store_true", help="Force rebuild places collection")
    parser.add_argument("--rebuild-events", action="store_true", help="Force rebuild events collection")
    parser.add_argument("--test", action="store_true", help="Test search functionality")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU (use CPU only)")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process per collection")
    
    args = parser.parse_args()
    
    print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
    print("\033[1m🧪 Vector Store Management (Incremental Sync)\033[0m", flush=True)
    print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
    
    try:
        kb = KnowledgeBase(use_gpu=not args.no_gpu)
        
        if args.stats:
            print("\n\033[1m📊 Database Statistics\033[0m", flush=True)
            stats = kb.get_stats()
            print(f"   Path: {stats['path']}", flush=True)
            print(f"   Total documents: {stats['total']}", flush=True)
            for col_name in [COLLECTION_PDF, COLLECTION_PLACES, COLLECTION_EVENTS]:
                col_stats = stats.get(col_name, {})
                status = col_stats.get("status", "unknown")
                count = col_stats.get("count", 0)
                print(f"   - {col_name}: {count} docs ({status})", flush=True)
        
        elif args.test:
            print("\n\033[1m🔍 Testing Vector Store...\033[0m", flush=True)
            stats = kb.get_stats()
            print(f"   Total documents: {stats['total']}", flush=True)
            
            if stats['total'] == 0:
                print("   \033[1;33m⚠️ Database is empty. Run sync first.\033[0m", flush=True)
            else:
                required_fields = {
                    "TurismoLisboa_OfficialGuide_PDF": ["source", "title", "url", "category", "content_hash", "indexed_at", "page"],
                    "VisitLisboa_Places": ["source", "title", "url", "category", "content_hash", "indexed_at"],
                    "VisitLisboa_Events": ["source", "title", "url", "category", "content_hash", "indexed_at"]
                }
                
                print("\n\033[1m📋 Metadata Validation by Collection:\033[0m", flush=True)
                
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
                            print(f"\n   \033[1;33m⚠️ {col_name}: Empty collection\033[0m", flush=True)
                            continue
                        
                        count = collection.count()
                        print(f"\n   \033[1m{col_name}\033[0m ({count} docs)", flush=True)
                        
                        sample_size = min(3, len(result["ids"]))
                        missing_fields = set()
                        empty_fields = set()
                        
                        for i in range(sample_size):
                            metadata = result["metadatas"][i]
                            for field in required_fields.get(source_name, []):
                                if field not in metadata:
                                    missing_fields.add(field)
                                else:
                                    val = metadata[field]
                                    if isinstance(val, (int, float)):
                                        continue
                                    elif not val or val in ["N/A", "Unknown", ""]:
                                        empty_fields.add(field)
                            
                            title = metadata.get('title', 'N/A')[:60]
                            url = metadata.get('url', 'N/A')[:50]
                            category = metadata.get('category', 'N/A')
                            print(f"      {i+1}. Title: \033[1;36m{title}\033[0m", flush=True)
                            print(f"         URL: {url}", flush=True)
                            print(f"         Category: {category}", flush=True)
                        
                        if missing_fields:
                            print(f"      \033[1;31m❌ Missing fields: {', '.join(missing_fields)}\033[0m", flush=True)
                        if empty_fields:
                            print(f"      \033[1;33m⚠️ Empty/Invalid fields: {', '.join(empty_fields)}\033[0m", flush=True)
                        if not missing_fields and not empty_fields:
                            print(f"      \033[1;32m✓ All metadata fields valid\033[0m", flush=True)
                            
                    except Exception as e:
                        print(f"\n   \033[1;31m❌ {col_name}: Error - {e}\033[0m", flush=True)
                
                print("\n\033[1m🔍 Search Quality Test:\033[0m", flush=True)
                test_queries = [
                    ("museums in Belém", "Should return places/PDF about Belém museums"),
                    ("traditional Portuguese food", "Should return restaurants"),
                    ("events this week", "Should return events"),
                    ("metro transport", "Should return transport info from PDF/places")
                ]
                
                for query, expected in test_queries:
                    print(f"\n   \033[1m📝 Query:\033[0m {query}", flush=True)
                    print(f"      Expected: {expected}", flush=True)
                    results = kb.search(query, k=3)
                    for i, doc in enumerate(results, 1):
                        title = doc.metadata.get('title', 'N/A')[:50]
                        source = doc.metadata.get('source', 'N/A')
                        score_indicator = "✓" if title != "N/A" and title != "Unknown" else "✗"
                        print(f"      {i}. [{score_indicator}] {title} ({source})", flush=True)
        
        else:
            result = kb.sync_all(
                rebuild_pdf=args.rebuild_all or args.rebuild_pdf,
                rebuild_places=args.rebuild_all or args.rebuild_places,
                rebuild_events=args.rebuild_all or args.rebuild_events,
                max_docs=args.max_docs
            )
            
            if result.get("has_more_work"):
                print("\n\033[1;33m⚠️  Exiting with code 2 (More work pending)\033[0m", flush=True)
                sys.exit(2)
                
    except Exception as e:
        print(f"\n\033[1;31m❌ CRITICAL ERROR: {e}\033[0m", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
