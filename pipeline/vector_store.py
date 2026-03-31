"""
Ad Intelligence Pipeline — Vector Store (Qdrant)

Stores ad JSON as searchable embeddings + indexed metadata.
Uses hybrid approach:
  - Rich text fields → embedded for semantic search
  - Structured fields → payload with indexed keys for O(1) filtering

For POC: stores video_link alongside embedding (no MongoDB).
For production: swap video_link with mongo_id.

Usage:
    from pipeline.vector_store import AdVectorStore
    store = AdVectorStore()
    store.add_ad(ad_json, video_link="https://...")
    results = store.search("emotional family ads", filters={"company_name": "Nike"})
"""

import json
import hashlib
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)

# --- Paths ---
QDRANT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "qdrant_db"
COLLECTION_NAME = "ad_intelligence"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, ~80MB, CPU-friendly
EMBEDDING_DIM = 384


class AdVectorStore:
    """
    Qdrant-backed vector store for ad intelligence JSON.

    Key design:
      - company_name and product_category are INDEXED payloads → O(1) filter
      - Rich text fields are embedded → semantic similarity search
      - Hybrid queries: hash lookup → then vector rank on subset
    """

    def __init__(self, persist_dir: str | None = None):
        persist_path = persist_dir or str(QDRANT_PERSIST_DIR)
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        # Local file-based Qdrant (no server needed)
        self.client = QdrantClient(path=persist_path)

        # Load embedding model (CPU, ~80MB, cached after first load)
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model ready")

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection + payload indexes if not already present."""
        collections = [c.name for c in self.client.get_collections().collections]

        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {COLLECTION_NAME}")

            # Create payload indexes for O(1) filtering
            # NOTE: Only effective when running Qdrant as a server.
            # Local file-based mode ignores indexes (still works, just O(n) scan).
            # To enable: docker run -p 6333:6333 qdrant/qdrant
            #            then use QdrantClient(url="http://localhost:6333")
            self._indexed_fields = [
                "company_name", "product_category", "industry",
                "ad_style", "tone", "medium", "primary_language", "product_name",
            ]
            try:
                for field_name in self._indexed_fields:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                logger.info(f"Payload indexes created ({len(self._indexed_fields)} fields)")
            except Exception:
                # Local mode doesn't support indexes — that's fine for POC
                logger.info("Payload indexes skipped (local mode — works at POC scale)")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' exists ({self.count()} ads)")

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_ad(self, ad_json: dict, video_link: str = "", ad_id: str | None = None) -> str:
        """
        Store a single ad's JSON into the vector store.

        Args:
            ad_json: Full 13-section ad intelligence JSON.
            video_link: URL/path to the source video (POC; production uses mongo_id).
            ad_id: Optional custom ID. If None, auto-generated.

        Returns:
            The ad_id used for storage.
        """
        # Build searchable document text
        document = self._build_document(ad_json)

        # Build payload (flat metadata for filtering)
        payload = self._build_payload(ad_json, video_link, document)

        # Generate embedding
        embedding = self.embedder.encode(document).tolist()

        # Generate ID (Qdrant uses UUIDs or unsigned ints — we'll use a hash-based UUID)
        if not ad_id:
            ad_id = self._generate_id(ad_json, video_link)

        point = PointStruct(
            id=ad_id,
            vector=embedding,
            payload=payload,
        )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point],
        )

        brand = payload.get("company_name", "Unknown")
        product = payload.get("product_name", "Unknown")
        logger.info(f"Stored ad: {brand} — {product} (id={ad_id[:12]}...)")
        return ad_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str = "",
        filters: dict | None = None,
        n_results: int = 10,
    ) -> list[dict]:
        """
        Hybrid search: semantic query + optional indexed filters.

        Args:
            query: Natural language search (e.g., "emotional family paint ads").
            filters: Exact match on indexed payload fields.
                     e.g., {"company_name": "Nike"}
                     e.g., {"company_name": ["Nike", "Adidas"]} for multi-match
            n_results: Max results to return.

        Returns:
            List of dicts: {id, score, payload}
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        # Pure filter search (no semantic query)
        if not query.strip():
            return self._filter_only_search(qdrant_filter, n_results)

        # Semantic search (with optional filter)
        query_embedding = self.embedder.encode(query).tolist()

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=n_results,
        )

        return [
            {
                "id": str(hit.id),
                "score": round(hit.score, 4),
                "payload": hit.payload or {},
            }
            for hit in response.points
        ]

    def search_by_brand(self, brand_name: str, n_results: int = 100) -> list[dict]:
        """Exact brand search — returns ALL ads for a brand."""
        return self.search(query="", filters={"company_name": brand_name}, n_results=n_results)

    def search_hybrid(
        self,
        query: str,
        brand: str | None = None,
        industry: str | None = None,
        tone: str | None = None,
        ad_style: str | None = None,
        n_results: int = 10,
    ) -> list[dict]:
        """
        Convenience method for common filter combos.

        Example:
            store.search_hybrid("family moments", brand="Benjamin Moore", tone="Playful")
        """
        filters = {}
        if brand:
            filters["company_name"] = brand
        if industry:
            filters["industry"] = industry
        if tone:
            filters["tone"] = tone
        if ad_style:
            filters["ad_style"] = ad_style

        return self.search(query=query, filters=filters or None, n_results=n_results)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of ads in the store."""
        info = self.client.get_collection(COLLECTION_NAME)
        return info.points_count

    def get_all_brands(self) -> list[str]:
        """List all unique brand names."""
        # Scroll through all points, collect unique brands
        brands = set()
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["company_name"],
            )
            for point in results:
                name = (point.payload or {}).get("company_name", "")
                if name:
                    brands.add(name)
            if offset is None:
                break
        return sorted(brands)

    def get_all_metadata_values(self) -> dict[str, set[str]]:
        """
        Get all unique values for key filterable fields.
        Useful for building Streamlit filter dropdowns.

        Returns:
            {"company_name": {"Nike", "Apple", ...}, "industry": {...}, ...}
        """
        fields = ["company_name", "product_category", "industry", "ad_style", "tone", "medium"]
        result = {f: set() for f in fields}

        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=fields,
            )
            for point in points:
                for f in fields:
                    val = (point.payload or {}).get(f, "")
                    if val:
                        result[f].add(val)
            if offset is None:
                break

        return result

    def delete_ad(self, ad_id: str):
        """Remove an ad by ID."""
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[ad_id],
        )
        logger.info(f"Deleted ad: {ad_id}")

    def clear_all(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self._ensure_collection()
        logger.info("Cleared all ads from vector store")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_document(self, ad_json: dict) -> str:
        """
        Concatenate rich text fields into a single string for embedding.
        These carry the most semantic meaning.
        """
        parts = []

        # Ad descriptions (highest signal)
        ad_desc = ad_json.get("ad_description", {})
        if ad_desc.get("detailed_description"):
            parts.append(ad_desc["detailed_description"])
        if ad_desc.get("short_summary"):
            parts.append(ad_desc["short_summary"])
        if ad_desc.get("creative_strategy"):
            parts.append(f"Creative strategy: {ad_desc['creative_strategy']}")

        # Text content
        text = ad_json.get("text_content", {})
        for field in ["headline", "tagline", "subheadline", "body_copy", "call_to_action"]:
            if text.get(field):
                parts.append(text[field])

        # Audio transcript
        audio = ad_json.get("audio", {})
        if audio.get("transcript"):
            parts.append(f"Transcript: {audio['transcript']}")

        # Visual analysis
        visual = ad_json.get("visual_analysis", {})
        if visual.get("scene_description"):
            parts.append(visual["scene_description"])
        if visual.get("people_description"):
            parts.append(visual["people_description"])

        # Classification themes + emotional appeal
        cls = ad_json.get("classification", {})
        if cls.get("themes"):
            themes = cls["themes"] if isinstance(cls["themes"], list) else [cls["themes"]]
            parts.append(f"Themes: {', '.join(themes)}")
        if cls.get("emotional_appeal"):
            parts.append(f"Emotional appeal: {cls['emotional_appeal']}")

        # Brand + Product context
        brand = ad_json.get("brand", {})
        product = ad_json.get("product", {})
        if brand.get("company_name"):
            parts.append(f"Brand: {brand['company_name']}")
        if product.get("product_name"):
            parts.append(f"Product: {product['product_name']}")
        if product.get("product_description"):
            parts.append(product["product_description"])

        document = " | ".join(p for p in parts if p)

        if not document:
            document = json.dumps(ad_json)
            logger.warning("No rich text fields found — embedding full JSON as fallback")

        return document

    def _build_payload(self, ad_json: dict, video_link: str, document: str) -> dict:
        """
        Extract flat payload for filtering + display.
        """
        brand = ad_json.get("brand", {})
        product = ad_json.get("product", {})
        text = ad_json.get("text_content", {})
        audio = ad_json.get("audio", {})
        visual = ad_json.get("visual_analysis", {})
        cls = ad_json.get("classification", {})
        source = ad_json.get("source", {})
        lang = ad_json.get("language", {})
        video = ad_json.get("video_analysis", {})
        ad_desc = ad_json.get("ad_description", {})

        # Flatten lists
        themes = cls.get("themes", [])
        if isinstance(themes, list):
            themes = ", ".join(themes)

        features = product.get("product_features", [])
        if isinstance(features, list):
            features = ", ".join(features)

        return {
            # Retrieval (POC)
            "video_link": video_link or "",

            # Searchable document text (stored for reference)
            "document": document,

            # Brand
            "company_name": brand.get("company_name", ""),
            "parent_company": brand.get("parent_company", ""),
            "sub_brand": brand.get("sub_brand", ""),
            "logo_detected": brand.get("logo_detected", False),

            # Product
            "product_name": product.get("product_name", ""),
            "product_category": product.get("product_category", ""),
            "product_subcategory": product.get("product_subcategory", ""),
            "model_or_variant": product.get("model_or_variant", ""),
            "product_features": features,

            # Text
            "tagline": text.get("tagline", ""),
            "headline": text.get("headline", ""),
            "call_to_action": text.get("call_to_action", ""),

            # Audio
            "has_speech": audio.get("has_speech", False),
            "has_music": audio.get("has_music", False),
            "voiceover_tone": audio.get("voiceover_tone", ""),

            # Visual
            "people_detected": visual.get("people_detected", False),
            "people_count": visual.get("people_count", 0),
            "scene_description": visual.get("scene_description", ""),

            # Classification (key indexed filters)
            "industry": cls.get("industry", ""),
            "ad_objective": cls.get("ad_objective", ""),
            "ad_style": cls.get("ad_style", ""),
            "target_audience": cls.get("target_audience", ""),
            "tone": cls.get("tone", ""),
            "themes": themes,
            "emotional_appeal": cls.get("emotional_appeal", ""),

            # Source
            "medium": source.get("medium", ""),
            "platform": source.get("platform", ""),

            # Language
            "primary_language": lang.get("primary_language", ""),

            # Video
            "duration_seconds": video.get("duration_seconds", 0.0),

            # Descriptions
            "short_summary": ad_desc.get("short_summary", ""),
            "detailed_description": ad_desc.get("detailed_description", ""),

            # Full JSON (for retrieval — gives everything back)
            "full_json": json.dumps(ad_json),
        }

    def _build_filter(self, filters: dict) -> Filter:
        """
        Convert simple dict to Qdrant Filter.

        Supports:
          {"company_name": "Nike"}                    → exact match
          {"company_name": ["Nike", "Adidas"]}        → match any
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions)

    def _filter_only_search(self, qdrant_filter: Filter | None, n_results: int) -> list[dict]:
        """Pure payload filter search — no vector similarity."""
        results, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qdrant_filter,
            limit=n_results,
            with_payload=True,
        )
        return [
            {
                "id": str(point.id),
                "score": 1.0,  # No similarity score for pure filter
                "payload": point.payload or {},
            }
            for point in results
        ]

    def _generate_id(self, ad_json: dict, video_link: str) -> str:
        """Generate a deterministic UUID from content."""
        brand = ad_json.get("brand", {}).get("company_name", "")
        product = ad_json.get("product", {}).get("product_name", "")
        tagline = ad_json.get("text_content", {}).get("tagline", "")
        content_key = f"{brand}|{product}|{tagline}|{video_link}"
        # Qdrant accepts string UUIDs
        return str(uuid.uuid5(uuid.NAMESPACE_URL, content_key))


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    store = AdVectorStore()
    print(f"Ads in store: {store.count()}")
    print(f"Brands: {store.get_all_brands()}")

    if store.count() > 0:
        results = store.search("emotional family ad")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['payload'].get('company_name')} — {r['payload'].get('product_name')}")
    else:
        print("No ads stored yet. Run test_vector_store.py to ingest sample ads.")