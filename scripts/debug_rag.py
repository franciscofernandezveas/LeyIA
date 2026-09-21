"""scripts/debug_rag.py — Debug: muestra qué chunks recupera el RAG.

Uso:
  python -m scripts.debug_rag "cual es el valor de una consulta"

Usa retrieve() de core.rag (mismos embeddings OpenAI 1536 y misma colección
que el agente en producción) — nunca chromadb crudo (embedding default 384).
"""
import sys

from core.rag import retrieve


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "cual es el valor de una consulta"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    docs = retrieve(query, k=k)
    print(f"\nQuery: {query!r} | k={k} | resultados: {len(docs)}\n")
    if not docs:
        print("(sin resultados — índice vacío o colección incorrecta)")
        return
    for i, d in enumerate(docs, 1):
        m = d.metadata
        print(f"── {i} [{m.get('tipo')}|sec {m.get('seccion')} | {m.get('seccion_titulo')}] ──")
        print(d.page_content[:350])
        print()


if __name__ == "__main__":
    main()
