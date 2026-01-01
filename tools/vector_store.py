# ==========================================================================
# Master Thesis - Vector Store Management (Incremental Sync)
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Manages the RAG (Retrieval-Augmented Generation) knowledge base.
#   Uses separate ChromaDB collections for different data sources with
#   incremental synchronization to avoid redundant processing.
# ==========================================================================

import sys
import os

# 🚀 CRITICAL: Force unbuffered output immediately to debug GitHub Actions hangs
sys.stdout.reconfigure(line_buffering=True)
print(f"\033[1m🚀 Starting vector_store.py at {os.getcwd()}...\033[0m", flush=True)

# Set environment variables BEFORE any heavy imports
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import json
import hashlib
import warnings
import argparse
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# Lazy imports for heavy libraries to prevent startup hangs
if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

# ==========================================================================
# Constants
# ==========================================================================
COLLECTION_PDF = "lisbon_pdf"
COLLECTION_PLACES = "lisbon_places"
COLLECTION_EVENTS = "lisbon_events"


def compute_content_hash(content: str) -> str:
    """Computes a SHA-256 hash of the content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def generate_doc_id(url: str, source: str) -> str:
    """Generates a stable document ID from URL and source."""
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
    return f"{source}_{url_hash}"


class KnowledgeBase:
    """Manages the RAG knowledge base with incremental synchronization."""
    
    def __init__(self, use_gpu: bool = True):
        """Initializes the KnowledgeBase with embedding model."""
        print(f"\033[1m📥 Initializing KnowledgeBase...\033[0m", flush=True)
        
        # Lazy import heavy libraries here
        print("   Importing AI libraries (this may take a moment)...", flush=True)
        global Chroma, HuggingFaceEmbeddings, Document, RecursiveCharacterTextSplitter, PyPDFLoader
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        from tqdm import tqdm
        self.tqdm = tqdm
        
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
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )
    
    def _get_existing_docs(self, collection_name: str) -> Dict[str, str]:
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
        try:
            vectorstore = self._get_collection(collection_name)
            vectorstore.delete_collection()
            print(f"   \033[1;33m🗑️ Deleted collection: {collection_name}\033[0m", flush=True)
        except Exception:
            pass

    def _extract_title(self, item: Dict[str, Any], source_tag: str) -> str:
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
            doc.metadata.update({
                "source": "TurismoLisboa_OfficialGuide_PDF",
                "title": f"{pdf_title} (p.{page_num + 1})",
                "url": f"{os.path.basename(pdf_path)}#page={page_num + 1}",
                "category": "Official Guide",
                "page": page_num + 1,
                "content_hash": compute_content_hash(doc.page_content),
                "indexed_at": datetime.now().isoformat()
            })
        
        print(f"   📊 Indexing {len(docs)} chunks...", flush=True)
        Chroma.from_documents(
            documents=docs, embedding=self.embeddings,
            collection_name=COLLECTION_PDF, persist_directory=self.vector_db_path,
            ids=doc_ids
        )
        print(f"   \033[1;32m✓ Indexed {len(docs)} PDF chunks\033[0m", flush=True)
        return {"status": "indexed", "added": len(docs)}

    def _sync_json_collection(
        self, collection_name: str, json_path: str, source_tag: str,
        force_rebuild: bool = False, max_docs: int = None
    ) -> Dict[str, int]:
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
            batch_size = 100  # Smaller batch size for safety
            
            print(f"   🔄 Indexing {len(docs_to_add)} documents...", flush=True)
            
            # Use tqdm if available and running interactively, else simple print
            iterator = range(0, len(docs_to_add), batch_size)
            if sys.stdout.isatty():
                iterator = self.tqdm(iterator, total=(len(docs_to_add) + batch_size - 1) // batch_size, desc="   Batch")

            for i in iterator:
                batch_docs = docs_to_add[i:i + batch_size]
                batch_ids = ids_to_add[i:i + batch_size]
                vectorstore.add_documents(batch_docs, ids=batch_ids)
                if not sys.stdout.isatty():
                    print(f"      Processed batch {i//batch_size + 1} ({len(batch_ids)} docs)", flush=True)
            
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
        return self._sync_json_collection(
            COLLECTION_PLACES, str(Config.PATH_VISIT_LISBOA_PLACES), "VisitLisboa_Places", force_rebuild, max_docs
        )
    
    def sync_events_collection(self, force_rebuild: bool = False, max_docs: int = None) -> Dict[str, int]:
        return self._sync_json_collection(
            COLLECTION_EVENTS, str(Config.PATH_VISIT_LISBOA_EVENTS), "VisitLisboa_Events", force_rebuild, max_docs
        )

    def sync_all(self, rebuild_pdf: bool = False, rebuild_places: bool = False, 
                 rebuild_events: bool = False, max_docs: int = None) -> Dict[str, Any]:
        print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
        print("\033[1m🔄 Vector Store Incremental Sync\033[0m", flush=True)
        if max_docs:
            print(f"\033[1m   (Max {max_docs} docs per collection)\033[0m", flush=True)
        print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
        
        results = {}
        has_more_work = False
        
        results["pdf"] = self.sync_pdf_collection(force_rebuild=rebuild_pdf)
        
        results["places"] = self.sync_places_collection(force_rebuild=rebuild_places, max_docs=max_docs)
        if results["places"].get("has_more_work"):
            has_more_work = True
        
        # If places took the quota, we still check events but maybe with 0 limit? 
        # Actually, let's allow events to run with the SAME limit, or remaining limit?
        # The user wants "PERFEITO". 
        # If we have a global timeout, we should probably respect max_docs strictly per run.
        # If places used 200 docs, and we have 200 limit, we should probably stop or let events run another 200?
        # The shell script says "max 200 docs per batch".
        # If we process 200 places AND 200 events, that's 400 docs. Might timeout.
        # Let's be conservative. If places used the quota, skip events for this run?
        # Or just let it run. Events are usually few.
        
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
    parser = argparse.ArgumentParser(description="Vector Store Management")
    parser.add_argument("--rebuild-all", action="store_true")
    parser.add_argument("--rebuild-pdf", action="store_true")
    parser.add_argument("--rebuild-places", action="store_true")
    parser.add_argument("--rebuild-events", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    
    args = parser.parse_args()
    
    print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
    print("\033[1m🧪 Vector Store CLI\033[0m", flush=True)
    print("\033[1m" + "=" * 60 + "\033[0m", flush=True)
    
    try:
        kb = KnowledgeBase(use_gpu=not args.no_gpu)
        
        if args.stats:
            stats = kb.get_stats()
            print(f"Total: {stats['total']}", flush=True)
            for k, v in stats.items():
                if k != "total" and k != "path":
                    print(f" - {k}: {v.get('count', 0)}", flush=True)
        
        elif args.test:
            print("Testing search...", flush=True)
            results = kb.search("museums", k=3)
            for doc in results:
                print(f" - {doc.metadata.get('title')}", flush=True)
        
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
