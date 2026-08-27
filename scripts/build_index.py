"""CLI para (re)indexar knowledge.md en ChromaDB."""
import argparse
from pathlib import Path

from core.config import KNOWLEDGE_PATH
from core.indexer import index_knowledge, parse_knowledge_markdown


def main():
    parser = argparse.ArgumentParser(description="Indexa knowledge.md en ChromaDB.")
    parser.add_argument("--path", default=None, help="Ruta alternativa al .md")
    parser.add_argument("--append", action="store_true", help="Agregar sin borrar la colección")
    parser.add_argument("--dry-run", action="store_true", help="Solo vista previa de chunks")
    args = parser.parse_args()

    path = Path(args.path) if args.path else KNOWLEDGE_PATH

    if args.dry_run:
        docs = parse_knowledge_markdown(path)
        for i, d in enumerate(docs, 1):
            print(f"\n── Chunk {i:02d} [{d.metadata['tipo']}|sec {d.metadata['seccion']}] ──")
            print(d.page_content[:250].replace("\n", " ⏎ ") + "…")
        print(f"\nTotal: {len(docs)} chunks")
        return

    n = index_knowledge(path=path, reset=not args.append)
    print(f"✅ {n} chunks indexados en chroma_db/")


if __name__ == "__main__":
    main()
