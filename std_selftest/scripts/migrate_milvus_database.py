"""Copy vectors from Milvus default.knowledge to the configured target database."""

import argparse
import asyncio
import sys
from pathlib import Path

from pymilvus import AsyncMilvusClient
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db.models import KnowledgeChunk
from app.db.session import AsyncSessionLocal
from app.services.vector_store import VectorStore


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-db",
        default="default",
        help="Source Milvus database name",
    )
    parser.add_argument(
        "--source-collection",
        default="knowledge",
        help="Source Milvus collection name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop the target collection before migration",
    )
    args = parser.parse_args()

    target = VectorStore()
    source = None
    try:
        if args.force:
            await target.reset()
        await target.ensure_collection()

        target_stats = await target.client.get_collection_stats(target.collection)
        if target_stats.get("row_count", 0) and not args.force:
            print(
                f"Target already has {target_stats['row_count']} rows. "
                "Use --force to replace it."
            )
            return 1

        source = AsyncMilvusClient(
            uri=settings.milvus_uri,
            db_name=args.source_db,
        )
        if not await source.has_collection(args.source_collection):
            print(
                f"Source collection {args.source_db}.{args.source_collection} "
                "does not exist."
            )
            return 1

        rows = await source.query(
            collection_name=args.source_collection,
            filter="",
            output_fields=["chunk_id", "category", "text", "vector"],
            limit=16384,
        )
        if not rows:
            print("Source collection is empty.")
            return 0

        async with AsyncSessionLocal() as db:
            result = await db.scalars(select(KnowledgeChunk.chunk_id))
            current_chunk_ids = set(result.all())

        records_by_chunk_id = {}
        for row in rows:
            chunk_id = row["chunk_id"]
            if chunk_id not in current_chunk_ids:
                continue
            records_by_chunk_id.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "category": row["category"],
                    "text": row["text"],
                    "vector": row["vector"],
                },
            )
        records = list(records_by_chunk_id.values())
        migrated_chunk_ids = {row["chunk_id"] for row in records}
        missing_chunk_ids = current_chunk_ids - migrated_chunk_ids

        if not records:
            print("No current MySQL chunk_id has a matching source vector.")
            return 1

        result = await target.client.insert(target.collection, records)
        target_ids = result.get("ids", [])
        await target.flush()

        async with AsyncSessionLocal() as db:
            for row, milvus_id in zip(records, target_ids, strict=False):
                item = await db.scalar(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.chunk_id == row["chunk_id"]
                    )
                )
                if item:
                    item.vector_status = "vectorized"
                    item.milvus_id = int(milvus_id)
            for chunk_id in missing_chunk_ids:
                item = await db.scalar(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.chunk_id == chunk_id
                    )
                )
                if item:
                    item.vector_status = "pending"
                    item.milvus_id = None
            await db.commit()

        print(
            f"Migrated {len(records)} current rows from "
            f"{args.source_db}.{args.source_collection} to "
            f"{settings.milvus_database}.{target.collection}; "
            f"{len(missing_chunk_ids)} rows marked pending."
        )
        return 0
    finally:
        if source is not None:
            await source.close()
        await target.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
