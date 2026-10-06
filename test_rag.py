# test_rag.py
from core.rag import retrieve

try:
    docs = retrieve("pensión de alimentos", k=3)
    print(f"OK: {len(docs)} documentos")
    for d in docs:
        print("-", d.page_content[:80].replace("\n", " "), "...")
except Exception as e:
    import traceback
    print("FALLA:", type(e).__name__, e)
    traceback.print_exc()
